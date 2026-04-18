import numpy as np
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
        pass
    
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
    pass