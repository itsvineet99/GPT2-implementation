import torch
import torch.nn as nn 

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        self.eps = 1e-5
        self.scale = nn.Linear(torch.ones(emb_dim))
        self.shift = nn.Linear(torch.zeros(emb_dim))

    def forward(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        var = torch.var(x, dim=-1,keepdim=True, unbiased=False)
        norm_x = (x-mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length,
                 num_heads, dropout, qkv_bias=False):
        super().__init__()

        assert (d_out % num_heads == 0), "d_out must be divisible num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.query_w = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.key_w = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.value_w = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_out, d_out)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        quries = self.query_w(x)
        keys = self.key_w(x)
        values = self.value_w(x)

        quries = quries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        quries = quries.transpose(1,2)
        keys = keys.transpose(1,2)
        values = values.transpose(1,2)

        attn_scores = quries @ keys.transpose(2,3)

        mask_bl = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill(mask_bl, -torch.inf)

        attn_weights = torch.softmax(
            attn_scores/keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        context_vector = (attn_weights @ values).transpose(1,2)
        context_vector = context_vector.contiguous().view(
            b, num_tokens, self.d_out
        )
        context_vector = self.out_proj(context_vector)

        return context_vector

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
                torch.sqrt(torch.tensor(2.0/torch.pi))
                * (x + 0.044715 * torch.pow(x, 3))
            ))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(cfg.emb_dim, 4*cfg.emb_dim),
            GELU(),
            nn.Linear(4*cfg.emb_dim, cfg.emb_dim)
        )

    def forward(self, x):
        return self.layers(x)
    

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.attention = MultiHeadAttention(
            d_in=cfg.emb_dim,
            d_out=cfg.emb_dim,
            context_length=cfg.context_length,
            num_heads=cfg.n_heads,
            dropout=cfg.drop_rate,
            qkv_bias=cfg.qkv_bias
        )
        self.ffn = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg.emb_dim)
        self.norm2 = LayerNorm(cfg.emb_dim)
        self.drop_shortcut = nn.Dropout(cfg.drop_rate)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attention(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.emb_dim)
        self.emb_drop = nn.Dropout(cfg.drop_rate)

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in cfg.n_layers]
        )

        self.final_norm = LayerNorm(cfg.emb_dim)
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape

        tok_emb = self.tok_emb(in_idx)
        pos_emb = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_emb + pos_emb
        x = self.emb_drop(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)

        logits = self.out_head(x)

        return logits
