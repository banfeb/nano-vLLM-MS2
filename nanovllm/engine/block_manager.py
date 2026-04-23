from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def _finalize_computed_blocks(self, seq: Sequence, num_computed_tokens: int):
        num_full_blocks = num_computed_tokens // self.block_size
        for i in range(num_full_blocks):
            block_id = seq.block_table[i]
            block = self.blocks[block_id]
            if block.hash != -1:
                continue
            token_ids = seq.block(i)
            prefix = self.blocks[seq.block_table[i - 1]].hash if i > 0 else -1
            h = self.compute_hash(token_ids, prefix)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block_id

    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        assert not seq.block_table
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.num_computed_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks-1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1

    def get_num_appendable_tokens(self, seq: Sequence) -> int:
        covered_tokens = len(seq.block_table) * self.block_size
        return covered_tokens - len(seq) + len(self.free_block_ids) * self.block_size

    def reserve_spec_append(self, seq: Sequence, num_tokens: int) -> list[int]:
        if num_tokens <= 0:
            return []

        num_required_blocks = (len(seq) + num_tokens + self.block_size - 1) // self.block_size - len(seq.block_table)
        if num_required_blocks <= 0:
            return []
        if len(self.free_block_ids) < num_required_blocks:
            raise RuntimeError("Insufficient KV cache capacity for speculative reservation.")

        new_block_ids: list[int] = []
        for _ in range(num_required_blocks):
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            new_block_ids.append(block_id)
        return new_block_ids

    def commit_spec_append(
        self,
        seq: Sequence,
        new_block_ids: list[int],
        num_computed_tokens: int,
    ):
        old_num_blocks = len(seq.block_table)
        required_num_blocks = (num_computed_tokens + self.block_size - 1) // self.block_size
        num_keep_blocks = max(0, required_num_blocks - old_num_blocks)
        seq.block_table.extend(new_block_ids[:num_keep_blocks])

        for block_id in new_block_ids[num_keep_blocks:]:
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)

        self._finalize_computed_blocks(seq, num_computed_tokens)
