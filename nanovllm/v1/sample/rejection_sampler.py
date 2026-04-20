import torch
import torch.nn as nn


class RejectionSampler(nn.Module):
    """
    A lightweight rejection sampler for speculative decoding.

    This implementation is intentionally simple and is sufficient for
    ngram-based drafting (where draft_probs is None).
    """

    def __init__(self, sampler: nn.Module):
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

        output_token_ids: list[list[int]] = []
        start = 0
        for req_idx, drafts in enumerate(draft_token_ids):
            req_output: list[int] = []
            rejected = False
            for pos, draft_token_id in enumerate(drafts):
                row_idx = start + pos
                token_id = int(draft_token_id)
                token_target_prob = float(target_probs[row_idx, token_id].item())
                if draft_probs is None:
                    token_draft_prob = 1.0
                else:
                    token_draft_prob = float(draft_probs[row_idx, token_id].item())

                if token_draft_prob <= 0:
                    is_accepted = False
                else:
                    accept_prob = min(1.0, token_target_prob / token_draft_prob)
                    is_accepted = bool(torch.rand(1, device=logits.device).item() <= accept_prob)

                if is_accepted:
                    req_output.append(token_id)
                    continue

                recovered_token_id = self._sample_recovered_token(
                    target_probs[row_idx], token_id, None if draft_probs is None else draft_probs[row_idx]
                )
                req_output.append(recovered_token_id)
                rejected = True
                break

            if not rejected:
                req_output.append(int(bonus_token_ids[req_idx].item()))

            output_token_ids.append(req_output)
            start += len(drafts)

        return output_token_ids

    @staticmethod
    def _sample_recovered_token(
        target_prob_row: torch.Tensor,
        draft_token_id: int,
        draft_prob_row: torch.Tensor | None,
    ) -> int:
        if draft_prob_row is None:
            recovered_probs = target_prob_row.clone()
            recovered_probs[draft_token_id] = 0
        else:
            recovered_probs = (target_prob_row - draft_prob_row).clamp_min_(0)

        prob_mass = recovered_probs.sum()
        if float(prob_mass.item()) <= 0:
            recovered_probs = target_prob_row
            prob_mass = recovered_probs.sum()
            if float(prob_mass.item()) <= 0:
                return int(torch.argmax(target_prob_row).item())

        recovered_probs = recovered_probs / prob_mass
        return int(torch.multinomial(recovered_probs, num_samples=1).item())


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
