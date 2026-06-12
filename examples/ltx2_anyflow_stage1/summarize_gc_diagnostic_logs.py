import argparse
import json
import math
from pathlib import Path


def read_text(path: Path):
    if not path.exists():
        return None
    return path.read_text(errors="replace")


def find_checkpoint(output_dir: Path):
    checkpoints = sorted(output_dir.glob("checkpoint-step_*/anyflow_wrapper.pt"))
    return checkpoints[0].parent if checkpoints else None


def main():
    parser = argparse.ArgumentParser(description="Summarize AnyFlow-LTX2 gradient-checkpointing diagnostic logs.")
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    output_dir = Path(args.output_dir)
    train_log = log_dir / "train.log"
    text = read_text(train_log)
    train_log_exists = text is not None
    text = text or ""

    checkpoint_dir = find_checkpoint(output_dir)
    step_log = output_dir / "anyflow_stage1_log_step_000001.json"
    step_log_exists = step_log.exists()

    flags = {
        "train_log_exists": train_log_exists,
        "checkpoint_error": "CheckpointError" in text,
        "saved_metadata": "saved metadata" in text,
        "recomputed_metadata": "recomputed metadata" in text,
        "cuda_oom": "CUDA out of memory" in text or "torch.OutOfMemoryError" in text,
        "dtype_mismatch": "same dtype" in text or "BFloat16 != float" in text,
        "smoke_finished": "Smoke finished" in text,
        "ok_message": "[OK]" in text,
        "checkpoint_exists": checkpoint_dir is not None,
        "anyflow_stage1_log_exists": step_log_exists,
    }

    log_values = {}
    if step_log_exists:
        try:
            payload = json.loads(step_log.read_text())
            for key in (
                "loss_total",
                "gradient_checkpointing_main_forward",
                "gradient_checkpointing_target_forward",
                "gradient_checkpointing_uncond_forward",
            ):
                log_values[key] = payload.get(key)
        except Exception as exc:
            log_values["read_error"] = repr(exc)

    if not train_log_exists:
        conclusion = "MISSING_LOGS"
    elif flags["cuda_oom"]:
        conclusion = "OOM"
    elif flags["dtype_mismatch"]:
        conclusion = "DTYPE_MISMATCH"
    elif flags["checkpoint_error"]:
        conclusion = "CHECKPOINT_ERROR"
    elif flags["checkpoint_exists"] and flags["anyflow_stage1_log_exists"]:
        conclusion = "SUCCESS"
    else:
        conclusion = "UNKNOWN_FAILURE"

    summary = {
        "log_dir": str(log_dir),
        "output_dir": str(output_dir),
        "train_log": str(train_log),
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir else None,
        "flags": flags,
        "anyflow_stage1_log_step_000001": log_values,
        "conclusion": conclusion,
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "gc_diagnostic_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
