import argparse
import os
import warnings

import accelerate
import torch
from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import LoadAudioWithTorchaudio, RouteByType, SequencialProcess, ToAbsolutePath
from diffsynth.diffusion import *
from diffsynth.pipelines.ltx2_audio_video import LTX2AudioVideoPipeline, ModelConfig

from anyflow_ltx2 import FlowMapSFTAudioVideoLoss, attach_anyflow_time_adapters, model_fn_ltx2_anyflow

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class LTX2AnyFlowTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None,
        model_id_with_origin_paths=None,
        tokenizer_path=None,
        trainable_models=None,
        lora_base_model=None,
        lora_target_modules="",
        lora_rank=32,
        lora_checkpoint=None,
        preset_lora_path=None,
        preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="flowmap_sft",
        anyflow_diffusion_ratio=0.25,
        anyflow_consistency_ratio=0.25,
        anyflow_reconstruction_weight=0.0,
    ):
        super().__init__()
        if not use_gradient_checkpointing:
            warnings.warn(
                "Gradient checkpointing is disabled. The training framework will enable it to reduce OOM risk."
            )
            use_gradient_checkpointing = True

        model_configs = self.parse_model_configs(
            model_paths,
            model_id_with_origin_paths,
            fp8_models=fp8_models,
            offload_models=offload_models,
            device=device,
        )
        tokenizer_config = (
            ModelConfig(model_id="google/gemma-3-12b-it-qat-q4_0-unquantized")
            if tokenizer_path is None
            else ModelConfig(tokenizer_path)
        )
        self.pipe = LTX2AudioVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
        )
        self.pipe.model_fn = model_fn_ltx2_anyflow
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
                "video_patchifier",
                "audio_patchifier",
            ),
            force_remove_params_shared=("audio_latents", "video_latents"),
            force_remove_params_nega=("audio_context", "video_context"),
        )
        self.switch_pipe_to_training_mode(
            self.pipe,
            trainable_models,
            lora_base_model,
            lora_target_modules,
            lora_rank,
            lora_checkpoint,
            preset_lora_path,
            preset_lora_model,
            task=task,
        )
        video_adapter_dim = getattr(getattr(self.pipe, "dit", None), "patchify_proj", None)
        video_adapter_dim = None if video_adapter_dim is None else video_adapter_dim.in_features
        audio_adapter_dim = getattr(getattr(self.pipe, "dit", None), "audio_patchify_proj", None)
        audio_adapter_dim = None if audio_adapter_dim is None else audio_adapter_dim.in_features
        attach_anyflow_time_adapters(self.pipe.dit, video_dim=video_adapter_dim, audio_dim=audio_adapter_dim)

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.anyflow_diffusion_ratio = anyflow_diffusion_ratio
        self.anyflow_consistency_ratio = anyflow_consistency_ratio
        self.anyflow_reconstruction_weight = anyflow_reconstruction_weight
        self.task_to_loss = {
            "flowmap_sft:data_process": lambda pipe, *args: args,
            "flowmap_sft": self.flowmap_sft_loss,
            "flowmap_sft:train": self.flowmap_sft_loss,
        }

    def flowmap_sft_loss(self, pipe, inputs_shared, inputs_posi, inputs_nega):
        return FlowMapSFTAudioVideoLoss(
            pipe,
            **inputs_shared,
            **inputs_posi,
            diffusion_ratio=self.anyflow_diffusion_ratio,
            consistency_ratio=self.anyflow_consistency_ratio,
            flowmap_reconstruction_weight=self.anyflow_reconstruction_weight,
        )

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

    def forward(self, data, inputs=None):
        if inputs is None:
            inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        return self.task_to_loss[self.task](self.pipe, *inputs)


def ltx2_anyflow_parser():
    parser = argparse.ArgumentParser(description="LTX2 AnyFlow flow-map pretraining entrypoint.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--frame_rate", type=float, default=24, help="Frame rate of the training videos.")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true")
    parser.add_argument("--anyflow_diffusion_ratio", type=float, default=0.25)
    parser.add_argument("--anyflow_consistency_ratio", type=float, default=0.25)
    parser.add_argument("--anyflow_reconstruction_weight", type=float, default=0.0)
    parser.set_defaults(task="flowmap_sft")
    return parser


if __name__ == "__main__":
    parser = ltx2_anyflow_parser()
    args = parser.parse_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
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
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=video_processor,
        special_operator_map={
            "input_audio": ToAbsolutePath(args.dataset_base_path)
            >> LoadAudioWithTorchaudio(
                num_frames=args.num_frames,
                time_division_factor=8,
                time_division_remainder=1,
                frame_rate=args.frame_rate,
            ),
            "in_context_videos": RouteByType(
                operator_map=[
                    (str, video_processor),
                    (list, SequencialProcess(video_processor)),
                ]
            ),
        },
    )
    model = LTX2AnyFlowTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
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
        anyflow_diffusion_ratio=args.anyflow_diffusion_ratio,
        anyflow_consistency_ratio=args.anyflow_consistency_ratio,
        anyflow_reconstruction_weight=args.anyflow_reconstruction_weight,
    )
    model_logger = ModelLogger(args.output_path, remove_prefix_in_ckpt=args.remove_prefix_in_ckpt)
    launcher_map = {
        "flowmap_sft:data_process": launch_data_process_task,
        "flowmap_sft": launch_training_task,
        "flowmap_sft:train": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
