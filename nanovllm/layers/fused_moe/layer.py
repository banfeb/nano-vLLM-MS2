import torch
import torch.nn as nn
from nanovllm.layers.fused_moe.router.fused_topk_router import (FusedTopKRouter)
from nanovllm.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)

# _ExpertLinear, _MergedExpertLinear, _ExpertMLP这3个类只是为了兼容模型加载逻辑
class _ExpertLinear(nn.Module):
    def __init__(self, out_features: int, in_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)


class _MergedExpertLinear(_ExpertLinear):
    def __init__(self, hidden_size: int, intermediate_size: int):
        self.intermediate_size = intermediate_size
        super().__init__(2 * intermediate_size, hidden_size)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int,
    ):
        shard_offset = loaded_shard_id * self.intermediate_size
        param.data.narrow(0, shard_offset, self.intermediate_size).copy_(loaded_weight)


class _ExpertMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_up_proj = _MergedExpertLinear(hidden_size, intermediate_size)
        self.down_proj = _ExpertLinear(hidden_size, intermediate_size)


class FusedMoE(nn.Module):
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        tp_size: int | None = None,
        activation: str = "silu",
    ):
        super().__init__()
        self.num_experts = num_experts
        self.global_num_experts = num_experts
        self.local_num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.tp_size = tp_size
        self.activation = activation
        for expert_id in range(num_experts):
            self.add_module(str(expert_id), _ExpertMLP(hidden_size, intermediate_size))
        self.quant_method = UnquantizedFusedMoEMethod(
            num_experts=num_experts,
            experts_per_token=top_k,
            hidden_dim=hidden_size,
            intermediate_size_per_partition=intermediate_size,
            num_local_experts=num_experts,
            activation=activation,
        )
        self.router = FusedTopKRouter(
            top_k=top_k,
            global_num_experts=self.global_num_experts,
            scoring_func="softmax",
        )

    def _get_expert(self, expert_id: int) -> _ExpertMLP:
        return getattr(self, str(expert_id))

    @property
    def w13_weight(self) -> torch.Tensor:
        """
            w13是所有专家门控FFN的gate, up权重矩阵的合并版本, shape(num_experts, hidden_size, 2 * intermediate_size)
        """
        experts = []
        for expert_id in range(self.num_experts):
            expert = self._get_expert(expert_id)
            experts.append(expert.gate_up_proj.weight.transpose(0, 1))
        return torch.stack(experts, dim=0).contiguous()

    @property
    def w2_weight(self) -> torch.Tensor:
        """
            w2是所有专家门控FFN的down权重矩阵的合并版本, shape(num_experts, intermediate_size, hidden_size)
        """
        experts = []
        for expert_id in range(self.num_experts):
            expert = self._get_expert(expert_id)
            experts.append(expert.down_proj.weight.transpose(0, 1))
        return torch.stack(experts, dim=0).contiguous()
        
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor
    ) -> torch.Tensor:
        topk_weights, topk_ids = self.router.select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )
        
        findal_hidden_states = self.quant_method.apply(
            layer=self,
            x=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
        )
        return findal_hidden_states
        
