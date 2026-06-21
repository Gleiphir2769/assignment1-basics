

from os import PathLike
import os
from pathlib import Path
import pickle

from cs336_basics.pretokenization_example import find_chunk_boundaries
import regex as re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT_RE = re.compile(PAT)

def _compile_special_pattern(special_tokens: list[str]):
    if not special_tokens:
        return None
    return re.compile("|".join(re.escape(tok) for tok in sorted(special_tokens, key=len, reverse=True)))

def _pretokenize_chunk(args):
    file_path, start, end, special_tokens = args
    counts = Counter()
    special_pattern = _compile_special_pattern(special_tokens)

    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    parts = special_pattern.split(chunk) if special_pattern else [chunk]
    for part in parts:
        counts.update(match.group(0).encode("utf-8") for match in PAT_RE.finditer(part))

    return counts


class BPETokenizer:
    def __init__(
        self,
        vocab_path: str | PathLike | None = None,
        vocab: dict[int, bytes] | None = None,
        merges: list[tuple[bytes, bytes]] | None = None,
        special_tokens: list[str] | None = None,
    ):
        self.vocab: dict[int, bytes] = {}
        self.token_to_id: dict[bytes, int] = {}
        self.merges: list[tuple[bytes, bytes]] = []
        self.merge_ranks: dict[tuple[bytes, bytes], int] = {}
        self.special_tokens = special_tokens or []
        self.vocab_path = Path(vocab_path) if vocab_path is not None else None
        self.loaded_vocab = False

        if vocab is not None:
            self.vocab = dict(vocab)
        if merges is not None:
            self.merges = list(merges)
        if vocab is not None or merges is not None:
            self.refresh_indexes()
        elif self.vocab_path is not None and self.vocab_path.exists():
            self.load_vocab(self.vocab_path)

    def refresh_indexes(self):
        self.token_to_id = {token_bytes: token_id for token_id, token_bytes in self.vocab.items()}
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}

    def save_vocab(self, vocab_path: str | PathLike | None = None):
        path = Path(vocab_path) if vocab_path is not None else self.vocab_path
        if path is None:
            raise ValueError("vocab_path is required when tokenizer was not initialized with one")

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {"vocab": self.vocab, "merges": self.merges, "special_tokens": self.special_tokens},
                f,
            )

    def load_vocab(self, vocab_path: str | PathLike):
        with open(vocab_path, "rb") as f:
            payload = pickle.load(f)

        self.vocab = payload["vocab"]
        self.merges = payload["merges"]
        self.special_tokens = payload.get("special_tokens", self.special_tokens)
        self.refresh_indexes()
        self.loaded_vocab = True

    def pre_tokenization(self, file_path: str | PathLike, special_tokens: list[str]):
        num_processes = os.cpu_count() or 1

        with open(file_path, "rb") as f:
            boundaries = find_chunk_boundaries(
                f,
                num_processes * 4,
                b"<|endoftext|>",
            )

        tasks = [
            (file_path, start, end, special_tokens)
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]

        pretoken_counts = Counter()
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            for chunk_counts in executor.map(_pretokenize_chunk, tasks):
                pretoken_counts.update(chunk_counts)

        return pretoken_counts

    def init_vocab(self, special_tokens: list[str]):
        vocab_id = 0
        self.vocab.clear()
        self.token_to_id.clear()
        self.merge_ranks.clear()
        self.merges.clear()
        self.special_tokens = special_tokens
        for st in special_tokens:
            self.vocab[vocab_id] = bytes(st, encoding="utf-8")
            vocab_id += 1
        for i in range(256):
            self.vocab[i+vocab_id] = bytes([i])

    def calc_pair_counts(self, word_counts):
        pair_counts = Counter()
        for word_count, count in word_counts.items():
            for pair in zip(word_count, word_count[1:]):
                pair_counts[pair] += count
        return pair_counts
    
    def merge_pair_in_tokens(self, tokens: tuple[bytes, ...], pair: tuple[bytes, bytes]):
        merged = []
        i = 0

        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
                merged.append(tokens[i] + tokens[i+1])
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        
        return tuple(merged)
    
    def build_word_counts(self, pretoken_counts):
        word_counts = Counter()
        for pt_key, pt_count in pretoken_counts.items():
            tokens = tuple(bytes([b]) for b in pt_key)
            word_counts[tokens] += pt_count
        return word_counts

    def train(
        self,
        input_path: str | PathLike,
        vocab_size: int,
        special_tokens: list[str],
        show_progress: bool = False,
        force_train: bool = False,
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        if self.loaded_vocab and not force_train:
            return self.vocab, self.merges

        self.init_vocab(special_tokens)
        pretoken_counts = self.pre_tokenization(input_path, special_tokens)

        word_counts = self.build_word_counts(pretoken_counts)

        num_merges = vocab_size - len(self.vocab)
        for _ in tqdm(range(num_merges), desc="Training BPE", disable=not show_progress):
            pair_counts = self.calc_pair_counts(word_counts)

            if not pair_counts:
                break

            best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
            self.merges.append(best_pair)
            self.vocab[len(self.vocab)] = best_pair[0] + best_pair[1]

            new_word_counts = Counter()
            for tokens, count in word_counts.items():
                new_tokens = self.merge_pair_in_tokens(tokens, best_pair)
                new_word_counts[new_tokens] += count
            word_counts = new_word_counts

        self.refresh_indexes()
        self.loaded_vocab = True

        if self.vocab_path is not None:
            self.save_vocab(self.vocab_path)

        return self.vocab, self.merges
    
    def decode(self, ids):
        data = b"".join(self.vocab[id] for id in ids)
        return data.decode("utf-8", errors="replace")
    
    def split_special_tokens(self, text: str):
        special_pattern = _compile_special_pattern(self.special_tokens)
        if special_pattern is None:
            yield text, False
            return

        last_end = 0
        for match in special_pattern.finditer(text):
            if match.start() > last_end:
                yield text[last_end:match.start()], False
            yield match.group(0), True
            last_end = match.end()

        if last_end < len(text):
            yield text[last_end:], False

    def apply_merges(self, tokens: tuple[bytes, ...]):
        while True:
            best_pair = None
            best_rank = None

            for pair in zip(tokens, tokens[1:]):
                rank = self.merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_pair = pair
                    best_rank = rank

            if best_pair is None:
                return tokens

            tokens = self.merge_pair_in_tokens(tokens, best_pair)

    def encode(self, text: str):
        ids = []

        for part, is_special in self.split_special_tokens(text):
            if is_special:
                ids.append(self.token_to_id[part.encode("utf-8")])
                continue

            for match in PAT_RE.finditer(part):
                pretoken = match.group(0).encode("utf-8")
                tokens = tuple(bytes([b]) for b in pretoken)
                tokens = self.apply_merges(tokens)
                ids.extend(self.token_to_id[token] for token in tokens)

        return ids

    def encode_iterable(self, iterable):
        for text in iterable:
            yield from self.encode(text)

if __name__ == "__main__":
    bpe = BPETokenizer("/home/daqige/code/cs336/assignment1-basics/cs336_basics/vocab/bpe_vocab")
    bpe.train(
        "/home/daqige/code/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt",
        1000,
        ["<|endoftext|>"],
        show_progress=True,
    )

    encoded = bpe.encode("Hello, World!")
    print(encoded)
    decoded = bpe.decode(encoded)
    print(decoded)
