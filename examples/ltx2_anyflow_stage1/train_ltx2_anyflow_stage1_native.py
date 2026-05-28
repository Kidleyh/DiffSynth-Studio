import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import accelerate
import torch
from accelerate import Accelerator
from tqdm import tqdm
from transformers.integrations.deepspeed import HfDeepSpeedConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import LoadAudioWithTorchaudio, RouteByType, SequencialProcess, ToAbsolutePath
from diffsynth.diffusion import *
from diffsynth.diffusion.logger import ModelLogger
from diffsynth.diffusion.runner import initialize_deepspeed_gradient_checkpointing, launch_data_process_task
from diffsynth.pipelines.ltx2_audio_video import LTX2AudioVideoPipeline, ModelConfig

from anyflow_ltx2_debug import collect_gradient_sanity
from anyflow_ltx2_lora import inject_lora_linear, looks_like_tiny_time_only_trainable, parse_name_filter, trainable_parameter_report
from anyflow_ltx2_model_wrapper import LTX2AnyFlowWrapper
from anyflow_ltx2_native_loss_adapter import anyflow_stage1_native_loss

os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from deepspeed.utils import safe_get_full_fp32_param
except Exception:  # pragma: no cover - deepspeed is optional outside native smoke.
    safe_get_full_fp32_param = None


class AnyFlowLTX2NativeTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None,
        model_id_with_origin_paths=None,
        tokenizer_path=None,
        trainable_models=None,
        lora_base_model=None,
        lora_target_modules="to_k,to_q,to_v,to_out.0",
        lora_rank=32,
        lora_alpha=None,
        lora_scale=None,
        lora_checkpoint=None,
        preset_lora_path=None,
        preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="anyflow_stage1:train",
        gate_init=0.25,
        audio_loss_weight=1.0,
        boundary_prob=0.5,
        fd_eps=1e-3,
        cfg_fused=False,
        cfg_scale=1.0,
        disable_time_weight=False,
        disable_adaptive_weight=False,
        allow_tiny_trainable_params=False,
    ):
        super().__init__()
        model_configs = self.parse_model_configs(
            model_paths,
            model_id_with_origin_paths,
            fp8_models=fp8_models,
            offload_models=offload_models,
            device=device,
        )
        tokenizer_config = ModelConfig(model_id="google/gemma-3-12b-it-qat-q4_0-unquantized") if tokenizer_path is None else ModelConfig(tokenizer_path)
        self.pipe = LTX2AudioVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
        )
        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.pipe = self.split_pipeline_units(
            task,
            self.pipe,
            trainable_models,
            lora_base_model,
            remove_unnecessary_params=True,
            loss_required_params=(
                "input_latents",
                "audio_input_latents",
                "video_latents",
                "audio_latents",
                "video_context",
                "audio_context",
                "video_positions",
                "audio_positions",
                "video_patchifier",
                "audio_patchifier",
                "use_gradient_checkpointing",
                "use_gradient_checkpointing_offload",
            ),
            force_remove_params_shared=("audio_latents", "video_latents"),
            force_remove_params_nega=("audio_context", "video_context"),
        )

        self.task = task
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.model_paths_raw = model_paths
        self.model_id_with_origin_paths_raw = model_id_with_origin_paths
        self.lora_base_model = lora_base_model
        self.lora_target_modules = lora_target_modules
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha if lora_alpha is not None else (lora_scale if lora_scale is not None else lora_rank)
        self.lora_checkpoint = lora_checkpoint
        self.gate_init = gate_init
        self.audio_loss_weight = audio_loss_weight
        self.boundary_prob = boundary_prob
        self.fd_eps = fd_eps
        self.cfg_fused = cfg_fused
        self.cfg_scale = cfg_scale
        self.disable_time_weight = disable_time_weight
        self.disable_adaptive_weight = disable_adaptive_weight
        self.gradient_sanity_checked = False
        self.trainable_report = None

        if not task.endswith(":data_process"):
            self._install_anyflow_wrapper_and_lora(allow_tiny_trainable_params)

        self.task_to_loss = {
            "anyflow_stage1:data_process": lambda pipe, *args: args,
            "anyflow_stage1": self.anyflow_loss,
            "anyflow_stage1:train": self.anyflow_loss,
        }

    def _install_anyflow_wrapper_and_lora(self, allow_tiny_trainable_params=False):
        if getattr(self.pipe, "dit", None) is None:
            raise RuntimeError("AnyFlow native train requires pipe.dit. Did you load transformer.safetensors?")
        wrapper = LTX2AnyFlowWrapper(self.pipe.dit, gate=self.gate_init, freeze_base=True)
        if self.lora_base_model is None and not allow_tiny_trainable_params:
            raise RuntimeError("AnyFlow native training requires --lora_base_model dit unless --allow_tiny_trainable_params is set.")
        if self.lora_base_model not in (None, "dit"):
            raise ValueError("AnyFlow native trainer currently supports --lora_base_model dit only.")
        if self.lora_base_model == "dit":
            updated = inject_lora_linear(
                wrapper.dit,
                rank=self.lora_rank,
                alpha=self.lora_alpha,
                name_filter=parse_name_filter(self.lora_target_modules),
            )
            print(f"injected AnyFlow LoRA into {updated} LTX2 DiT linear layers", flush=True)
            if updated == 0:
                raise RuntimeError("No LoRA target modules matched --lora_target_modules.")
        self.pipe.dit = wrapper
        self.trainable_report = trainable_parameter_report(self)
        self._print_trainable_report()
        if not allow_tiny_trainable_params:
            if self.trainable_report["trainable"] < 1_000_000 or looks_like_tiny_time_only_trainable(self.trainable_report["names"]):
                raise RuntimeError(
                    "Trainable parameter set is too small for AnyFlow Stage 1. "
                    "Use --lora_base_model dit --lora_rank 256 for real training, "
                    "or pass --allow_tiny_trainable_params for debugging only."
                )

    def _print_trainable_report(self):
        report = self.trainable_report or trainable_parameter_report(self)
        print(f"total params: {report['total']:,}", flush=True)
        print(f"trainable params: {report['trainable']:,}", flush=True)
        print(f"trainable ratio: {report['ratio']:.8f}", flush=True)
        print("trainable module name sample:", flush=True)
        for name in report["name_sample"]:
            print(f"  {name}", flush=True)

    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_images"] = [data["video"][0]]
                inputs_shared["input_images_indexes"] = [0]
                inputs_shared["input_images_strength"] = 1.0
            else:
                inputs_shared[extra_input] = data[extra_input]
        return inputs_shared

    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        inputs_shared = {
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            "frame_rate": data.get("frame_rate", 24),
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "video_patchifier": self.pipe.video_patchifier,
            "audio_patchifier": self.pipe.audio_patchifier,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega

    def anyflow_loss(self, pipe, inputs_shared, inputs_posi, inputs_nega):
        loss, logs = anyflow_stage1_native_loss(
            pipe,
            pipe.dit,
            inputs_shared,
            inputs_posi,
            audio_loss_weight=self.audio_loss_weight,
            boundary_prob=self.boundary_prob,
            fd_eps=self.fd_eps,
            cfg_fused=self.cfg_fused,
            cfg_scale=self.cfg_scale,
            use_time_weight=not self.disable_time_weight,
            use_adaptive_weight=not self.disable_adaptive_weight,
        )
        self.latest_anyflow_logs = logs
        return loss

    def forward(self, data, inputs=None):
        if inputs is None:
            inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        return self.task_to_loss[self.task](self.pipe, *inputs)

    def anyflow_config(self, args=None):
        return {
            "native_trainer": True,
            "task": self.task,
            "model_paths": self.model_paths_raw,
            "model_id_with_origin_paths": self.model_id_with_origin_paths_raw,
            "lora_base_model": self.lora_base_model,
            "lora_target_modules": self.lora_target_modules,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "gate_init": self.gate_init,
            "audio_loss_weight": self.audio_loss_weight,
            "boundary_prob": self.boundary_prob,
            "fd_eps": self.fd_eps,
            "cfg_fused": self.cfg_fused,
            "cfg_scale": self.cfg_scale,
            "time_weight": "beta(2,1.5)" if not self.disable_time_weight else "disabled",
            "adaptive_loss": {"enabled": not self.disable_adaptive_weight},
            "gradient_sanity_checked": self.gradient_sanity_checked,
            "frozen_unused_r_adaln_linear": True,
            "trainable_without_grad_policy": "allow" if getattr(args, "allow_trainable_without_grad", False) else "raise",
        }


def save_anyflow_sidecar(accelerator, model, output_path, step, optimizer=None, args=None):
    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    ckpt = Path(output_path) / f"checkpoint-step_{step:06d}"
    trainable = {}
    for name, param in unwrapped.named_parameters():
        if not param.requires_grad:
            continue
        full_param = None
        if safe_get_full_fp32_param is not None:
            try:
                full_param = safe_get_full_fp32_param(param)
            except Exception:
                full_param = None
        if full_param is None:
            full_param = param.detach()
        trainable[name] = full_param.detach().cpu()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        ckpt.mkdir(parents=True, exist_ok=True)
        torch.save(trainable, ckpt / "anyflow_wrapper.pt")
        optimizer_payload = {
            "note": "Native DeepSpeed/Accelerate optimizer state is not gathered into the AnyFlow sidecar.",
            "global_step": step,
        }
        if optimizer is not None and "deepspeed" not in str(type(optimizer)).lower():
            try:
                optimizer_payload = optimizer.state_dict()
            except Exception as exc:
                optimizer_payload["state_dict_error"] = repr(exc)
        torch.save(optimizer_payload, ckpt / "optimizer.pt")
        (ckpt / "training_state.json").write_text(json.dumps({"global_step": step}, indent=2))
        (ckpt / "anyflow_config.json").write_text(json.dumps(unwrapped.anyflow_config(args), indent=2))


def save_gradient_sanity(output_path, step, report):
    path = Path(output_path) / f"gradient_sanity_step_{step:06d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))


def launch_anyflow_native_training_task(accelerator: Accelerator, dataset, model, model_logger, args):
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        collate_fn=lambda x: x[0],
        num_workers=args.dataset_num_workers,
    )
    uses_deepspeed = getattr(accelerator.state, "deepspeed_plugin", None) is not None
    if uses_deepspeed:
        model.pipe.device = accelerator.device
    else:
        model.to(device=accelerator.device)
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    initialize_deepspeed_gradient_checkpointing(accelerator)

    global_step = 0
    for _epoch_id in range(args.num_epochs):
        for data in tqdm(dataloader):
            with accelerator.accumulate(model):
                loss = model({}, inputs=data) if dataset.load_from_cache else model(data)
                accelerator.backward(loss)
                global_step += 1
                unwrapped = accelerator.unwrap_model(model)
                if not unwrapped.gradient_sanity_checked:
                    report = collect_gradient_sanity(unwrapped)
                    if accelerator.is_main_process:
                        print(
                            "gradient sanity: "
                            f"with_grad={report['trainable_with_grad_count']} "
                            f"without_grad={report['trainable_without_grad_count']} "
                            f"top20={report['grad_norm_top20']}",
                            flush=True,
                        )
                        if args.save_gradient_sanity:
                            save_gradient_sanity(args.output_path, global_step, report)
                    if report["trainable_zero_grad_names"] and accelerator.is_main_process:
                        warnings.warn(
                            "Some trainable tensors have zero gradients: "
                            + ", ".join(report["trainable_zero_grad_names"][:20])
                        )
                    if report["trainable_without_grad_names"] and not args.allow_trainable_without_grad:
                        raise RuntimeError(
                            "Trainable tensors without gradients were found. "
                            f"First names: {report['trainable_without_grad_names'][:20]}"
                        )
                    unwrapped.gradient_sanity_checked = True
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model_logger.on_step_end(accelerator, model, args.save_steps, loss=loss)
                if args.save_steps is not None and global_step % args.save_steps == 0:
                    save_anyflow_sidecar(accelerator, model, args.output_path, global_step, optimizer, args)
                if args.max_steps is not None and global_step >= args.max_steps:
                    save_anyflow_sidecar(accelerator, model, args.output_path, global_step, optimizer, args)
                    return
    save_anyflow_sidecar(accelerator, model, args.output_path, global_step, optimizer, args)


def ltx2_anyflow_native_parser():
    parser = argparse.ArgumentParser(description="Native-style AnyFlow-LTX2 Stage 1 trainer.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--frame_rate", type=float, default=24, help="Frame rate of the training videos.")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true")
    parser.add_argument("--lora_alpha", type=float, default=None)
    parser.add_argument("--lora_scale", type=float, default=None)
    parser.add_argument("--gate_init", type=float, default=0.25)
    parser.add_argument("--audio_loss_weight", type=float, default=1.0)
    parser.add_argument("--boundary_prob", type=float, default=0.5)
    parser.add_argument("--fd_eps", type=float, default=1e-3)
    parser.add_argument("--cfg_fused", action="store_true")
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--disable_time_weight", action="store_true")
    parser.add_argument("--disable_adaptive_weight", action="store_true")
    parser.add_argument("--allow_tiny_trainable_params", action="store_true")
    parser.add_argument("--allow_trainable_without_grad", action="store_true")
    parser.add_argument("--save_gradient_sanity", action="store_true", default=True)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.set_defaults(task="anyflow_stage1:train")
    return parser


def build_dataset(args):
    video_processor = UnifiedDataset.default_video_operator(
        base_path=args.dataset_base_path,
        max_pixels=args.max_pixels,
        height=args.height,
        width=args.width,
        height_division_factor=32,
        width_division_factor=32,
        num_frames=args.num_frames,
        time_division_factor=8,
        time_division_remainder=1,
        frame_rate=args.frame_rate,
        fix_frame_rate=True,
    )
    return UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=video_processor,
        special_operator_map={
            "input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudioWithTorchaudio(
                num_frames=args.num_frames,
                time_division_factor=8,
                time_division_remainder=1,
                frame_rate=args.frame_rate,
            ),
            "in_context_videos": RouteByType(operator_map=[
                (str, video_processor),
                (list, SequencialProcess(video_processor)),
            ]),
        },
    )


if __name__ == "__main__":
    parser = ltx2_anyflow_native_parser()
    args = parser.parse_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    if getattr(accelerator.state, "deepspeed_plugin", None) is not None:
        HfDeepSpeedConfig(accelerator.state.deepspeed_plugin.deepspeed_config)

    dataset = build_dataset(args)
    model = AnyFlowLTX2NativeTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_scale=args.lora_scale,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if (args.initialize_model_on_cpu or args.enable_model_cpu_offload) else accelerator.device,
        gate_init=args.gate_init,
        audio_loss_weight=args.audio_loss_weight,
        boundary_prob=args.boundary_prob,
        fd_eps=args.fd_eps,
        cfg_fused=args.cfg_fused,
        cfg_scale=args.cfg_scale,
        disable_time_weight=args.disable_time_weight,
        disable_adaptive_weight=args.disable_adaptive_weight,
        allow_tiny_trainable_params=args.allow_tiny_trainable_params,
    )
    model_logger = ModelLogger(args.output_path, remove_prefix_in_ckpt=args.remove_prefix_in_ckpt)
    launcher_map = {
        "anyflow_stage1:data_process": launch_data_process_task,
        "anyflow_stage1": launch_anyflow_native_training_task,
        "anyflow_stage1:train": launch_anyflow_native_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
