import torch


class FlowMapEulerSchedulerForLTX2:
    """Continuous-time Euler scheduler for AnyFlow LTX2 stage-1 sampling."""

    def __init__(self, num_train_timesteps: int = 1000, device=None, dtype=torch.float32):
        self.num_train_timesteps = num_train_timesteps
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype
        self.timesteps = None

    def set_timesteps(self, num_inference_steps: int, device=None, dtype=None):
        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be >= 1")
        device = self.device if device is None else torch.device(device)
        dtype = self.dtype if dtype is None else dtype
        self.timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype
        return self.timesteps

    @staticmethod
    def _as_broadcast_time(value, sample: torch.Tensor):
        value = torch.as_tensor(value, device=sample.device, dtype=sample.dtype)
        if value.ndim == 0:
            value = value.reshape(1)
        while value.ndim < sample.ndim:
            value = value.view(*value.shape, 1)
        return value

    def add_noise(self, x: torch.Tensor, noise: torch.Tensor, t) -> torch.Tensor:
        t = self._as_broadcast_time(t, x).clamp(0.0, 1.0)
        return (1.0 - t) * x + t * noise.to(device=x.device, dtype=x.dtype)

    def step(self, model_output: torch.Tensor, sample: torch.Tensor, t, r) -> torch.Tensor:
        t = self._as_broadcast_time(t, sample).clamp(0.0, 1.0)
        r = self._as_broadcast_time(r, sample).clamp(0.0, 1.0)
        return sample - (t - r) * model_output.to(device=sample.device, dtype=sample.dtype)

