import torch


class LoRALinear(torch.nn.Module):
    def __init__(self, base, rank=16, alpha=None):
        super().__init__()
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha if alpha is not None else rank)
        self.lora_A = torch.nn.Linear(base.in_features, self.rank, bias=False)
        self.lora_B = torch.nn.Linear(self.rank, base.out_features, bias=False)
        torch.nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        torch.nn.init.zeros_(self.lora_B.weight)
        for param in self.base.parameters():
            param.requires_grad = False

    @property
    def scale(self):
        return self.alpha / self.rank

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(x)) * self.scale


def parse_name_filter(value):
    if value is None:
        return ("attn", "ff", "proj")
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    return tuple(value)


def inject_lora_linear(module, rank=16, alpha=None, name_filter=("attn", "ff", "proj"), prefix=""):
    name_filter = parse_name_filter(name_filter)
    updated = 0
    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        full_hit = any(token in full_name for token in name_filter)
        if isinstance(child, torch.nn.Linear) and full_hit:
            setattr(module, child_name, LoRALinear(child, rank=rank, alpha=alpha))
            updated += 1
        else:
            updated += inject_lora_linear(child, rank=rank, alpha=alpha, name_filter=name_filter, prefix=full_name)
    return updated


def trainable_parameter_report(module, sample_limit=30):
    total = sum(param.numel() for param in module.parameters())
    trainable = sum(param.numel() for param in module.parameters() if param.requires_grad)
    names = [name for name, param in module.named_parameters() if param.requires_grad]
    return {
        "total": total,
        "trainable": trainable,
        "ratio": trainable / max(total, 1),
        "names": names,
        "name_sample": names[:sample_limit],
    }


def looks_like_tiny_time_only_trainable(names):
    if not names:
        return True
    allowed = ("gate", "r_adaln")
    return all(any(token in name for token in allowed) for name in names)

