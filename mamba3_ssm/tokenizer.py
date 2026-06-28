"""Tokenizers for Mamba-3: Char-level (default) and BPE."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class CharTokenizer:
    """Simple character-level tokenizer."""

    def __init__(self, text: str = ""):
        chars = sorted(set(text))
        self.chars = ["<pad>"] + chars  # index 0 = padding
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> List[int]:
        return [self.stoi.get(c, 0) for c in text]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos.get(i, "") for i in ids)

    def save(self, path: Path):
        data = {"chars": self.chars, "type": "char"}
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> "CharTokenizer":
        data = json.loads(path.read_text(encoding="utf-8"))
        tok = CharTokenizer()
        tok.chars = data["chars"]
        tok.stoi = {c: i for i, c in enumerate(tok.chars)}
        tok.itos = {i: c for i, c in enumerate(tok.chars)}
        return tok


class BPETokenizer:
    """BPE tokenizer backed by HuggingFace tokenizers library."""

    def __init__(self, vocab_size: int = 8192):
        self.vocab_size = vocab_size
        self._tok = None  # lazy init

    def _lazy_init(self):
        if self._tok is not None:
            return
        from tokenizers import Tokenizer as HFTokenizer
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPre
        from tokenizers.decoders import ByteLevel as ByteLevelDec
        from tokenizers.trainers import BpeTrainer
        self._hf = HFTokenizer(BPE(unk_token="<unk>"))
        self._hf.pre_tokenizer = ByteLevelPre(add_prefix_space=False)
        self._hf.decoder = ByteLevelDec()
        self._trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["<pad>", "<unk>"],
            initial_alphabet=ByteLevelPre.alphabet(),
        )

    def train(self, texts: List[str]):
        self._lazy_init()
        self._hf.train_from_iterator(texts, trainer=self._trainer)

    def encode(self, text: str) -> List[int]:
        self._lazy_init()
        out = self._hf.encode(text)
        return out.ids

    def decode(self, ids: List[int]) -> str:
        self._lazy_init()
        return self._hf.decode(ids)

    def save(self, path: Path):
        self._lazy_init()
        self._hf.save(str(path))

    @staticmethod
    def load(path: Path) -> "BPETokenizer":
        from tokenizers import Tokenizer as HFTokenizer
        tok = BPETokenizer()
        tok._hf = HFTokenizer.from_file(str(path))
        tok.vocab_size = tok._hf.get_vocab_size()
        return tok


def load_tokenizer(data_dir, vocab_size: int = 256) -> Tuple[object, str]:
    """Load tokenizer from data_dir (str or Path). Returns (tokenizer, type_str)."""
    data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir
    bpe_path = data_dir / "tokenizer_bpe.json"
    if bpe_path.exists():
        return BPETokenizer.load(bpe_path), "bpe"

    # Fall back to char
    char_path = data_dir / "tokenizer.json"
    if char_path.exists():
        return CharTokenizer.load(char_path), "char"

    # Create a minimal char tokenizer with byte fallback
    tok = CharTokenizer()
    tok.chars = [chr(i) for i in range(vocab_size)]
    tok.stoi = {c: i for i, c in enumerate(tok.chars)}
    tok.itos = {i: c for i, c in enumerate(tok.chars)}
    return tok, "byte"
