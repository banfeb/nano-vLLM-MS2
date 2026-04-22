import torch
import torch.nn as nn
import triton
import triton.language as tl
from nanovllm.layers.sampler import Sampler

PLACEHOLDER_TOKEN_ID = -1


@triton.jit
def _sample_recovered_tokens_kernel(
    output_token_ids_ptr,  # [num_tokens]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    draft_probs_ptr,  # [num_tokens, vocab_size]
    target_probs_ptr,  # [num_tokens, vocab_size]
    q_ptr,
    vocab_size,
    PADDED_VOCAB_SIZE: tl.constexpr,
    NO_DRAFT_PROBS: tl.constexpr,
):
    req_idx = tl.program_id(0).to(tl.int32)
    start_idx = tl.zeros((), dtype=tl.int32)
    if req_idx > 0:
        start_idx = tl.load(cu_num_draft_tokens_ptr + req_idx - 1).to(tl.int32)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx).to(tl.int32)
    num_draft_tokens = end_idx - start_idx

    pos = tl.program_id(1)
    if pos >= num_draft_tokens:
        return

    vocab_offset = tl.arange(0, PADDED_VOCAB_SIZE)
    row_idx = start_idx + pos
    if NO_DRAFT_PROBS:
        draft_token_id = tl.load(draft_token_ids_ptr + row_idx)
        prob = tl.load(
            target_probs_ptr + row_idx * vocab_size + vocab_offset,
            mask=((vocab_offset < vocab_size) & (vocab_offset != draft_token_id)),
            other=0.0,
        )
    else:
        draft_prob = tl.load(
            draft_probs_ptr + row_idx * vocab_size + vocab_offset,
            mask=vocab_offset < vocab_size,
            other=0.0,
        )
        target_prob = tl.load(
            target_probs_ptr + row_idx * vocab_size + vocab_offset,
            mask=vocab_offset < vocab_size,
            other=0.0,
        )
        prob = tl.maximum(target_prob - draft_prob, 0.0)

    q = tl.load(
        q_ptr + req_idx * vocab_size + vocab_offset,
        mask=vocab_offset < vocab_size,
        other=float("-inf"),
    )
    recovered_id = tl.argmax(prob / q, axis=-1)
    tl.store(output_token_ids_ptr + row_idx, recovered_id)


@triton.jit(do_not_specialize=["max_spec_len"])
def _rejection_random_sample_kernel(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    draft_probs_ptr,  # [num_tokens, vocab_size]
    target_probs_ptr,  # [num_tokens, vocab_size]
    bonus_token_ids_ptr,  # [batch_size]
    recovered_token_ids_ptr,  # [num_tokens]
    uniform_probs_ptr,  # [num_tokens]
    max_spec_len,
    vocab_size,
    NO_DRAFT_PROBS: tl.constexpr,
):
    req_idx = tl.program_id(0).to(tl.int32)
    start_idx = tl.zeros((), dtype=tl.int32)
    if req_idx > 0:
        start_idx = tl.load(cu_num_draft_tokens_ptr + req_idx - 1).to(tl.int32)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx).to(tl.int32)
    num_draft_tokens = end_idx - start_idx

    rejected = False
    for pos in range(num_draft_tokens):
        if not rejected:
            draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
            draft_token_id_i32 = draft_token_id.to(tl.int32)

            if NO_DRAFT_PROBS:
                draft_prob = 1.0
            else:
                draft_prob = tl.load(
                    draft_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id_i32
                )

            target_prob = tl.load(
                target_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id_i32
            )
            uniform_prob = tl.load(uniform_probs_ptr + start_idx + pos)

            if draft_prob > 0 and target_prob / draft_prob >= uniform_prob:
                token_id = draft_token_id
            else:
                rejected = True
                token_id = tl.load(recovered_token_ids_ptr + start_idx + pos)

            tl.store(
                output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos,
                token_id,
            )

    if not rejected:
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        tl.store(
            output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens,
            bonus_token_id,
        )


@triton.jit
def _uniform_probs_kernel(
    output_ptr,
    num_tokens,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_tokens
    uniform = tl.rand(seed, offsets)
    uniform = tl.maximum(uniform, 1e-7)
    tl.store(output_ptr + offsets, uniform, mask=mask)


class RejectionSampler(nn.Module):
    """
    A lightweight rejection sampler for speculative decoding.

    This implementation is intentionally simple and is sufficient for
    ngram-based drafting (where draft_probs is None).
    """

    def __init__(self, sampler: Sampler):
        super().__init__()
        self.sampler = sampler

    def forward(
        self,
        draft_token_ids: list[list[int]],
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        draft_probs: torch.Tensor | None = None,
    ) -> list[list[int]]:
        """
        Args:
            draft_token_ids:
                Draft tokens proposed by drafter. Shape: [bs, <=k]
            logits:
                Flattened target logits in the same layout as vLLM:
                [sum(num_draft_tokens) + bs, vocab_size].
                First part is per-draft-position logits, second part is
                one bonus-token logit row for each request.
            temperatures:
                Sampling temperatures. Shape: [bs]
            draft_probs:
                Optional draft model probabilities for each draft position.
                Shape: [sum(num_draft_tokens), vocab_size]. For ngram, keep None.
        Returns:
            output token ids for each request:
            accepted draft prefix + recovered token (if rejected), or
            all draft tokens + one bonus token (if all accepted).
        """
        batch_size = len(draft_token_ids)
        if batch_size == 0:
            return []
        if temperatures.ndim != 1 or temperatures.size(0) != batch_size:
            raise ValueError("temperatures must have shape [batch_size]")

        num_draft_tokens = [len(x) for x in draft_token_ids]
        total_num_draft_tokens = sum(num_draft_tokens)
        expected_rows = total_num_draft_tokens + batch_size

        if logits.ndim != 2 or logits.size(0) != expected_rows:
            raise ValueError(
                f"logits must have shape [{expected_rows}, vocab_size], "
                f"got {tuple(logits.shape)}"
            )
        if draft_probs is not None:
            expected_shape = (total_num_draft_tokens, logits.size(-1))
            if tuple(draft_probs.shape) != expected_shape:
                raise ValueError(
                    "draft_probs shape mismatch: "
                    f"expected {expected_shape}, got {tuple(draft_probs.shape)}"
                )

        bonus_logits = logits[total_num_draft_tokens:]
        bonus_token_ids = self.sampler(bonus_logits, temperatures).to(torch.int64)

        if total_num_draft_tokens == 0:
            return [[int(bonus_token_ids[i].item())] for i in range(batch_size)]

        target_logits = logits[:total_num_draft_tokens].to(torch.float32)
        token_temperatures = _expand_batch_to_tokens(
            temperatures.to(torch.float32),
            num_draft_tokens,
            total_num_draft_tokens,
            target_logits.device,
        )
        target_logits = target_logits.div(token_temperatures.unsqueeze(-1))
        target_probs = torch.softmax(target_logits, dim=-1, dtype=torch.float32)

        max_spec_len = max(num_draft_tokens)
        cu_num_draft_tokens = torch.tensor(
            num_draft_tokens,
            dtype=torch.int32,
            device=logits.device,
        ).cumsum(dim=0)
        flat_draft_token_ids = torch.tensor(
            [token_id for row in draft_token_ids for token_id in row],
            dtype=torch.int64,
            device=logits.device,
        )
        uniform_probs = generate_uniform_probs(
            total_num_draft_tokens,
            logits.device,
        )
        recovered_token_ids = sample_recovered_tokens(
            num_draft_tokens,
            cu_num_draft_tokens,
            flat_draft_token_ids,
            draft_probs,
            target_probs,
            logits.device,
        )

        output_token_ids = torch.full(
            (batch_size, max_spec_len + 1),
            PLACEHOLDER_TOKEN_ID,
            dtype=torch.int64,
            device=logits.device,
        )
        no_draft_probs = draft_probs is None
        draft_probs_for_kernel = (
            torch.empty_like(target_probs) if no_draft_probs else draft_probs
        )

        _rejection_random_sample_kernel[(batch_size,)](
            output_token_ids,
            cu_num_draft_tokens,
            flat_draft_token_ids,
            draft_probs_for_kernel,
            target_probs,
            bonus_token_ids.to(torch.int64),
            recovered_token_ids,
            uniform_probs,
            max_spec_len,
            target_probs.size(-1),
            NO_DRAFT_PROBS=no_draft_probs,
        )

        output_token_ids_cpu = output_token_ids.cpu().tolist()
        return [
            [token_id for token_id in row if token_id != PLACEHOLDER_TOKEN_ID]
            for row in output_token_ids_cpu
        ]


def generate_uniform_probs(
    num_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    if num_tokens == 0:
        return torch.empty((0,), dtype=torch.float32, device=device)
    if device.type != "cuda":
        raise RuntimeError("Triton uniform sampling requires CUDA device.")

    uniform_probs = torch.empty(
        (num_tokens,),
        dtype=torch.float32,
        device=device,
    )
    seed = int(
        torch.randint(
            0,
            2**31 - 1,
            (1,),
            device="cpu",
            dtype=torch.int64,
        ).item()
    )
    block_size = 1024
    grid = (triton.cdiv(num_tokens, block_size),)
    _uniform_probs_kernel[grid](
        uniform_probs,
        num_tokens,
        seed,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return uniform_probs


def sample_recovered_tokens(
    num_draft_tokens: list[int],
    cu_num_draft_tokens: torch.Tensor,
    draft_token_ids: torch.Tensor,
    draft_probs: torch.Tensor | None,
    target_probs: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if device.type != "cuda":
        raise RuntimeError("Triton recovered token sampling requires CUDA device.")

    batch_size = len(num_draft_tokens)
    num_tokens = draft_token_ids.numel()
    vocab_size = target_probs.size(-1)
    q = torch.empty(
        (batch_size, vocab_size),
        dtype=torch.float32,
        device=device,
    )
    q.exponential_()

    recovered_token_ids = torch.empty_like(draft_token_ids)
    draft_probs_for_kernel = target_probs if draft_probs is None else draft_probs
    max_spec_len = max(num_draft_tokens)
    _sample_recovered_tokens_kernel[(batch_size, max_spec_len)](
        recovered_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs_for_kernel,
        target_probs,
        q,
        vocab_size,
        triton.next_power_of_2(vocab_size),
        NO_DRAFT_PROBS=draft_probs is None,
    )
    return recovered_token_ids


def _expand_batch_to_tokens(
    values: torch.Tensor,
    num_tokens_per_request: list[int],
    total_num_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    expanded = torch.empty(total_num_tokens, dtype=values.dtype, device=device)
    start = 0
    for idx, count in enumerate(num_tokens_per_request):
        if count > 0:
            expanded[start : start + count] = values[idx]
            start += count
    return expanded
