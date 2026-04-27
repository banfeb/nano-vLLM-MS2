import numpy as np
import os
from numba import get_num_threads, jit, njit, prange, set_num_threads

class NgramProposer:
    def __init__(
        self,
        prompt_lookup_min: int = 1,
        prompt_lookup_max: int = 3,
        num_speculative_tokens: int = 2,
        max_model_len: int = None,
        max_num_seqs: int = None,
    ):
        assert max_model_len is not None, "max_model_len must be specified"
        assert max_num_seqs is not None, "max_num_seqs must be specified"
        self.min_n = prompt_lookup_min
        self.max_n = prompt_lookup_max
        self.k = num_speculative_tokens
        self.max_model_len = max_model_len
        self.valid_ngram_draft = np.zeros((max_num_seqs, self.k), dtype=np.int32)
        self.valid_ngram_num_drafts = np.zeros((max_num_seqs), dtype=np.int32)
        # NgramProposer.propose 只在rank = 0 上执行
        self.num_tokens_threshold = 8192
        cpu_count = os.cpu_count()
        if cpu_count:
            self.num_numba_thread_available = min(8, cpu_count // 2)
        else:
            self.num_numba_thread_available = 1

        self.propose(
            np.zeros((1024,), dtype=np.int32),
            np.zeros((1024, max_model_len), dtype=np.int32),
        )
        
    def propose(
        self, 
        num_tokens_no_spec: np.ndarray,
        token_ids_cpu: np.ndarray,
    ) -> list[list[int]]:
        # num_tokens_no_spec: (batch_size) 每个请求不包含speculative tokens的token数
        # token_ids_cpu: (batch_size, max_model_len)
        num_requests = int(num_tokens_no_spec.shape[0])
        if token_ids_cpu.shape[0] != num_requests:
            raise ValueError(
                "token_ids_cpu batch size must match num_tokens_no_spec, "
                f"got {token_ids_cpu.shape[0]} and {num_requests}."
            )
        valid_ngram_requests = []
        for i, num_tokens in enumerate(num_tokens_no_spec):
            # 空上下文或超长上下文都不做 speculative drafting
            if num_tokens <= 0:
                continue
            if num_tokens >= self.max_model_len:
                continue
            valid_ngram_requests.append(i)
        draft_token_ids = self.batch_propose(
            num_requests,
            valid_ngram_requests,
            num_tokens_no_spec,
            token_ids_cpu,
        )
        return draft_token_ids
    
    def batch_propose(
        self, 
        num_requests: int,
        valid_ngram_requests: list,
        num_tokens_no_spec: np.ndarray,
        token_ids_cpu: np.ndarray,
    ) -> list[list[int]]:
        draft_token_ids: list[list[int]] = []
        valid_ngram_request_set = set(valid_ngram_requests)

        if num_ngram_requests := len(valid_ngram_requests):
            original_num_numba_threads = get_num_threads()
            total_tokens = np.sum(num_tokens_no_spec)
            if total_tokens >= self.num_tokens_threshold:
                final_num_threads = max(
                    1, min(self.num_numba_thread_available, num_ngram_requests)
                )
                set_num_threads(final_num_threads)
            else:
                set_num_threads(1)

            batch_propose_numba(
                valid_ngram_requests,
                num_tokens_no_spec,
                token_ids_cpu,
                self.min_n,
                self.max_n,
                self.max_model_len,
                self.k,
                self.valid_ngram_draft,
                self.valid_ngram_num_drafts,
            )

            # Restore original number of threads.
            set_num_threads(original_num_numba_threads)

        for i in range(num_requests):
            if i in valid_ngram_request_set and self.valid_ngram_num_drafts[i] > 0:
                draft_token_ids.append(
                    self.valid_ngram_draft[i, : self.valid_ngram_num_drafts[i]].tolist()
                )
            else:
                draft_token_ids.append([])

        return draft_token_ids
    
    
    
@njit(parallel=True)
def batch_propose_numba(
    vaild_ngram_requests: list,
    num_tokens_no_spec: np.ndarray,
    token_ids_cpu: np.ndarray,
    min_n: int,
    max_n: int,
    max_model_len: int,
    k: int,
    valid_ngram_draft: np.ndarray,
    valid_ngram_num_drafts: np.ndarray,
):
    for i in prange(len(vaild_ngram_requests)):
        request_id = vaild_ngram_requests[i]
        num_tokens = num_tokens_no_spec[request_id]
        origin_tokens = token_ids_cpu[request_id][:num_tokens]
        proposed_tokens = _find_longest_matched_ngram_and_propose_tokens(
            origin_tokens, min_n, max_n, max_model_len, k
        )
        valid_ngram_draft[request_id][:len(proposed_tokens)] = proposed_tokens
        valid_ngram_num_drafts[request_id] = len(proposed_tokens)


@jit(nopython=True)
def _find_longest_matched_ngram_and_propose_tokens(
    origin_tokens: np.ndarray,
    min_ngram: int,
    max_ngram: int,
    max_model_len: int,
    k: int,
) -> np.ndarray:
    total_token = origin_tokens.shape[0]
    if total_token < min_ngram:
        return np.empty((0,), dtype=origin_tokens.dtype)

    k = min(k, max_model_len - total_token)
    if k <= 0:
        return np.empty((0,), dtype=origin_tokens.dtype)

    tokens = origin_tokens[::-1]

    lps = np.zeros(max_ngram, dtype=np.int32)

    longest_ngram = 0
    position = 0

    prev_lps = 0
    i = 1
    while i < total_token:
        if tokens[prev_lps] == tokens[i]:

            prev_lps += 1
            if prev_lps >= longest_ngram:
                longest_ngram = prev_lps
                position = i
            if i < max_ngram:
                lps[i] = prev_lps
            if prev_lps == max_ngram:
                prev_lps = lps[max_ngram - 1]
            i += 1
        elif prev_lps != 0:
            prev_lps = lps[prev_lps - 1]
        else:
            i += 1

    if longest_ngram < min_ngram:
        return np.empty((0,), dtype=origin_tokens.dtype)

    start_position = total_token - 1 - position + longest_ngram
    k = min(k, total_token - start_position)
    return origin_tokens[start_position : start_position + k]
