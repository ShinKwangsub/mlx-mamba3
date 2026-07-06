import json
from dataclasses import dataclass


@dataclass
class CharTokenizer:
    """Tiny deterministic character tokenizer for local smoke tests."""

    vocab: list[str]
    unk_token: str = "<unk>"

    def __post_init__(self):
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        if self.unk_token not in self.token_to_id:
            raise ValueError(f"unk_token {self.unk_token!r} must be in vocab")
        self.unk_id = self.token_to_id[self.unk_token]

    @classmethod
    def from_text(cls, text: str, special_tokens: tuple[str, ...] = ("<unk>",)):
        chars = sorted(set(text))
        vocab = list(special_tokens)
        vocab.extend(ch for ch in chars if ch not in special_tokens)
        return cls(vocab=vocab)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        return [self.token_to_id.get(ch, self.unk_id) for ch in text]

    def decode(self, ids: list[int] | tuple[int, ...]) -> str:
        chars = []
        for idx in ids:
            if idx < 0 or idx >= len(self.vocab):
                chars.append(self.unk_token)
            else:
                chars.append(self.vocab[idx])
        return "".join(chars)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"vocab": self.vocab, "unk_token": self.unk_token}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(vocab=data["vocab"], unk_token=data.get("unk_token", "<unk>"))
