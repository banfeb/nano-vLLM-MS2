from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.is_spec_decoding = config.speculative_config is not None
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def reserve_spec_decode(
        self,
        seqs: list[Sequence],
        draft_token_ids: list[list[int]],
    ) -> tuple[list[list[int]], list[dict[str, list[int] | int]]]:
        reserved_draft_token_ids: list[list[int]] = []
        reservations: list[dict[str, list[int] | int]] = []
        for seq, seq_draft_token_ids in zip(seqs, draft_token_ids):
            max_num_appendable_tokens = self.block_manager.get_num_appendable_tokens(seq)
            reserved_num_draft_tokens = min(len(seq_draft_token_ids), max_num_appendable_tokens)
            seq_draft_token_ids = seq_draft_token_ids[:reserved_num_draft_tokens]
            new_block_ids = self.block_manager.reserve_spec_append(seq, reserved_num_draft_tokens)
            reserved_draft_token_ids.append(seq_draft_token_ids)
            reservations.append({
                "draft_len": reserved_num_draft_tokens,
                "new_block_ids": new_block_ids,
            })
        return reserved_draft_token_ids, reservations

    def postprocess_spec_decode(
        self,
        seqs: list[Sequence],
        token_ids: list[list[int]],
        draft_token_ids: list[list[int]],
        reservations: list[dict[str, list[int] | int]],
    ) -> int:
        num_tokens: int = 0
        for seq, seq_token_ids, seq_draft_token_ids, reservation in zip(seqs, token_ids, draft_token_ids, reservations):
            old_len = len(seq)
            accepted_draft_tokens = 0
            for draft_token_id, token_id in zip(seq_draft_token_ids, seq_token_ids):
                if draft_token_id != token_id:
                    break
                accepted_draft_tokens += 1

            num_appended_accepted_drafts = 0
            for token_idx, token_id in enumerate(seq_token_ids):
                num_tokens += 1
                seq.append_token(token_id)
                if token_idx < accepted_draft_tokens:
                    num_appended_accepted_drafts += 1
                if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens >= seq.max_tokens:
                    seq.status = SequenceStatus.FINISHED
                    break

            num_computed_tokens = old_len + num_appended_accepted_drafts
            self.block_manager.commit_spec_append(
                seq,
                reservation["new_block_ids"],
                num_computed_tokens,
            )
            seq.num_computed_tokens = num_computed_tokens

            if seq.is_finished:
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
        return num_tokens

    def postprocess(self, seqs: list[Sequence], token_ids: list[int] | list[list[int]]) -> int:
        for seq, token_id in zip(seqs, token_ids):
            seq.num_computed_tokens = len(seq)
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens >= seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
        return len(seqs)
