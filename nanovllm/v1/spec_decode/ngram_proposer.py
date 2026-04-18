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
        # 决定是否启用多线程的参数(如果有tp需要重新考虑)
        # TODO: 之后完成spec_decode的tp并行后的线程参数调整
        self.num_tokens_threshold = 8192
        cpu_count = os.cpu_count()
        if cpu_count:
            self.num_numba_thread_available = min(8, cpu_count // 2)
        else:
            self.num_numba_thread_available = 1

        self.propose(
            [[]] * 1024,
            np.zeros((1024,), dtype=np.int32),
            np.zeros((1024, max_model_len), dtype=np.int32),
        )
        
    def propose(
        self, 
        sampled_token_ids: list[list[int]],
        num_tokens_no_spec: np.ndarray,
        token_ids_cpu: np.ndarray,
    ) -> list[list[int]]:
        valid_ngram_requests = []
        for i, sampled_ids in enumerate(sampled_token_ids):
            num_sampled_ids = len(sampled_ids)
            # 当request没有采样时，不进行speculative decoding
            if not num_sampled_ids:
                continue
            num_tokens = num_tokens_no_spec[i]
            if num_tokens >= self.max_model_len:
                continue
            valid_ngram_requests.append(i)
        draft_token_ids = self.batch_propose(
            len(sampled_token_ids),
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
            if i in valid_ngram_requests and self.valid_ngram_num_drafts[i] > 0:
                draft_token_ids.append(
                    self.valid_ngram_draft[i, : self.valid_ngram_num_drafts[i]].tolist()
                )
            else:
                draft_token_ids.append([])

        return draft_token_ids
    
    def load_model(self, *args, **kwargs):
        pass
    
    
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
    """
    Find the longest n-gram which matches the suffix of the given tokens
    whose length is within [min_ngram, max_ngram] (inclusive).

    If found, we will extract k right after the matched ngram.
    """
    # Do not generate draft tokens is context is shorter than minimum n-gram
    total_token = origin_tokens.shape[0]
    if total_token < min_ngram:
        return np.empty((0,), dtype=origin_tokens.dtype)

    # Do not generate draft tokens beyond the max model length.
    k = min(k, max_model_len - total_token)
    if k <= 0:
        return np.empty((0,), dtype=origin_tokens.dtype)

    # Flip tokens, and the goal become to find longest ngram
    # on the rightmost position which matches the prefix with
    # length [min_n, max_n] (inclusive).
    tokens = origin_tokens[::-1]

    # Longest prefix (not including itself) which is a suffix of
    # the current position.
    #   lps[i] = max{v, where tokens[0:v] == tokens[i+1-v:i+1]}
    #
    # As ngram is capped by max_ngram to save memory, we only need to
    # store lps for the first max_ngram prefix.
    lps = np.zeros(max_ngram, dtype=np.int32)

    longest_ngram = 0
    position = 0

    # lps[0] always equal to 0, we start with index 1
    prev_lps = 0
    i = 1
    while i < total_token:
        # tokens[:prev_lps] is the longest prefix as a suffix of tokens[:i]
        if tokens[prev_lps] == tokens[i]:
            # Token match: tokens[:prev_lps+1] is the longest prefix as
            # a suffix of tokens[:i+1]
            prev_lps += 1
            # Check if we found a longer valid ngram.
            #
            # Update position when longest_ngram matched prev_lps,
            # as we want to get the target n-gram of the earliest position
            # in the original tokens (i.e.
            # latest position in the reversed tokens)
            if prev_lps >= longest_ngram:
                longest_ngram = prev_lps
                position = i
            if i < max_ngram:
                # Store LPS for the first max_ngram prefix
                lps[i] = prev_lps
            if prev_lps == max_ngram:
                # When prev_lps reached max_ngram, update prev_lps
                # to lps[max_ngram-1] to avoid matching ngram
                # longer than max_ngram
                prev_lps = lps[max_ngram - 1]
            i += 1
        elif prev_lps != 0:
            # Token mismatch: try the second-longest prefix
            # among all suffix of tokens[:i],
            # which is the longest prefix of tokens[:prev_lps]
            prev_lps = lps[prev_lps - 1]
        else:
            # Token mismatch, and no more prefix (except empty string)
            # as a suffix of tokens[:i]
            i += 1

    if longest_ngram < min_ngram:
        # No valid ngram is found
        return np.empty((0,), dtype=origin_tokens.dtype)

    # Flip the position back, so in origin_tokens,
    # origin_tokens[total_token-1-position:total_token-1-position+longest_ngram]
    # is the matched ngram, so we should start drafting tokens from
    # total_token-1-position+longest_ngram
    start_position = total_token - 1 - position + longest_ngram
    k = min(k, total_token - start_position)
    return origin_tokens[start_position : start_position + k]