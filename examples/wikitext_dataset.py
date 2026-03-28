"""WikiText-103 dataset with GPT-2 tokenizer, cached to disk."""

import os
import torch
from torch.utils.data import Dataset

from megatron.core.datasets.utils import Split

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class WikiTextDataset(Dataset):
    """Pre-tokenised WikiText-103 chunks (GPT-2 tokenizer, cached to disk).

    First run tokenizes the full dataset and saves to examples/.data_cache/.
    Subsequent runs load from cache in < 1 second.
    """

    _CACHE_DIR = os.path.join(ROOT_DIR, "examples", ".data_cache")

    def __init__(self, seq_len: int, split: str = "train"):
        cache_file = os.path.join(self._CACHE_DIR, f"wikitext103_{split}_gpt2.pt")

        self._ensure_cache_ready(cache_file, split)
        cached = torch.load(cache_file, weights_only=True)
        all_ids = cached["token_ids"]
        self.vocab_size = cached["vocab_size"].item()

        chunk_len = seq_len + 1
        n_chunks = len(all_ids) // chunk_len
        self.data = all_ids[: n_chunks * chunk_len].view(n_chunks, chunk_len)
        self.seq_len = seq_len
        self.split_name = split
        self.split = {
            "train": Split.train,
            "validation": Split.valid,
            "test": Split.test,
        }[split]

    @classmethod
    def _ensure_cache_ready(cls, cache_file: str, split: str) -> None:
        if os.path.exists(cache_file):
            return

        os.makedirs(cls._CACHE_DIR, exist_ok=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0 and not os.path.exists(cache_file):
                cls._build_cache_file(cache_file, split)
            torch.distributed.barrier()
        elif not os.path.exists(cache_file):
            cls._build_cache_file(cache_file, split)

    @staticmethod
    def _build_cache_file(cache_file: str, split: str) -> None:
        from datasets import load_dataset
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)

        all_ids_list = []
        batch_texts = []
        for row in ds:
            text = row["text"].strip()
            if text:
                batch_texts.append(text)
            if len(batch_texts) >= 1000:
                enc = tokenizer(batch_texts, add_special_tokens=False)
                for ids in enc["input_ids"]:
                    all_ids_list.extend(ids)
                batch_texts = []
        if batch_texts:
            enc = tokenizer(batch_texts, add_special_tokens=False)
            for ids in enc["input_ids"]:
                all_ids_list.extend(ids)

        token_ids = torch.tensor(all_ids_list, dtype=torch.long)
        torch.save(
            {
                "token_ids": token_ids,
                "vocab_size": torch.tensor(tokenizer.vocab_size),
            },
            cache_file,
        )

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        chunk = self.data[idx]
        return {
            "tokens": chunk[:-1],
            "labels": chunk[1:],
            "loss_mask": torch.ones(self.seq_len, dtype=torch.float32),
            "position_ids": torch.arange(self.seq_len, dtype=torch.long),
        }
