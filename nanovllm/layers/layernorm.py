import torch
from torch import nn


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def _rms_forward_2d(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def _add_rms_forward_2d(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """传入x可能为2d或者3d, 在这个接口统一reshape为2d, 避免调用@torch.compile是重复编译"""
        input_shape = x.shape
        x = x.reshape(-1, input_shape[-1])
        x = self._rms_forward_2d(x)
        return x.reshape(input_shape)

    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """传入x和residual可能为2d或者3d, 在这个接口统一reshape为2d, 避免调用@torch.compile是重复编译"""
        input_shape = x.shape
        x = x.reshape(-1, input_shape[-1])
        residual = residual.reshape(-1, input_shape[-1])
        x, residual = self._add_rms_forward_2d(x, residual)
        return x.reshape(input_shape), residual.reshape(input_shape)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
