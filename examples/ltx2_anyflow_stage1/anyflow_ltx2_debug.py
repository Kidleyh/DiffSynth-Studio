from collections import defaultdict

import torch


def _prefix_for_name(name):
    parts = name.split(".")
    if "lora_A" in name or "lora_B" in name:
        return "lora"
    if "r_adaln" in name:
        return "r_adaln"
    if name.endswith(".gate") or ".gate" in name:
        return "gate"
    return ".".join(parts[:2]) if len(parts) >= 2 else parts[0]


def collect_gradient_sanity(model):
    without_grad = []
    zero_grad = []
    with_nonzero_grad = []
    grad_norms = []
    by_prefix = defaultdict(int)
    total_trainable_tensors = 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        total_trainable_tensors += 1
        by_prefix[_prefix_for_name(name)] += int(param.numel())
        if param.grad is None:
            without_grad.append(name)
            continue
        grad_norm = float(param.grad.detach().float().norm().cpu())
        if grad_norm == 0.0:
            zero_grad.append(name)
        else:
            with_nonzero_grad.append(name)
            grad_norms.append([name, grad_norm])

    grad_norms.sort(key=lambda item: item[1], reverse=True)
    return {
        "total_trainable_tensors": total_trainable_tensors,
        "trainable_with_grad_count": total_trainable_tensors - len(without_grad),
        "trainable_without_grad_count": len(without_grad),
        "trainable_with_nonzero_grad_count": len(with_nonzero_grad),
        "trainable_zero_grad_count": len(zero_grad),
        "trainable_without_grad_names": without_grad,
        "trainable_zero_grad_names": zero_grad,
        "grad_norm_top20": grad_norms[:20],
        "trainable_param_count_by_prefix": dict(sorted(by_prefix.items())),
    }

