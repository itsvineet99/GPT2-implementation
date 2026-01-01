from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 50257
    context_length: int = 1024
    emb_dim: int = 768
    n_heads: int = 12
    n_layers: int = 12
    drop_rate: float = 0.1
    qkv_bias: bool = False

    @classmethod
    def GPT_124M(cls):
        return cls(
            emb_dim=768,
            n_heads=12,
            n_layers=12)
    
    @classmethod
    def GPT2_355M(cls):
        """Returns configuration for the 355M parameter model (GPT-2 Medium)"""
        return cls(
            emb_dim=1024,
            n_heads=16,
            n_layers=24
        )
