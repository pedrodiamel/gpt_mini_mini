import torch
import torch.nn as nn
from torch.nn import functional as F


__all__ = ["GPTmm", "gptmm"]


class Head(nn.Module):
    """One head of self-attention"""

    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # [k]_(B,T,H)
        q = self.query(x)  # [q]_(B,T,H)
        v = self.value(x)  # [v]_(B,T,H)

        # compute attention score
        wei = q @ k.transpose(-2, -1) * C**-0.5  # [wei]_(B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # [wei]_(B,T,T)
        wei = F.softmax(wei, dim=-1)  # [wei]_(B,T,T)
        wei = self.dropout(wei)  # [wei]_(B,T,T)

        # perform the weihted aggregation of the values
        v = self.value(x)  # [v]_(B,T,H)
        out = wei @ v  # [out]_(B,T,H)

        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        self.wo = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)  # [x]_(B,T,E)
        x = self.dropout(self.wo(x))  # [x]_(B,T,E)
        return x


class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """A transformer block"""

    def __init__(self, n_embd, n_heads, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_heads
        self.sa_head = MultiHeadAttention(
            n_heads, head_size, n_embd, block_size, dropout
        )  # i.e. 4 heads of 8-dimensional self-attention
        self.ff_head = FeedForward(n_embd, dropout)  # E \to E
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa_head(self.ln1(x))  # [x]_(B,T,E)
        x = x + self.ff_head(self.ln2(x))  # [x]_(B,T,E)
        return x


class GPTmm(nn.Module):
    def __init__(self, n_layers, vocab_size, n_embd, n_heads, block_size, dropout=0.2):
        super().__init__()
        # f(x) \in {B,T,C}
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)  # [V,E]
        self.position_embedding_table = nn.Embedding(block_size, n_embd)  # [T,E]
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(n_embd, n_heads, block_size, dropout) for _ in range(n_layers)]
        )  # E \to E

        self.block_size = block_size
        self.vocab_size = vocab_size
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)  # E \to V

        # init all weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        # [idx]_(B,T)
        # [targets]_(B,T)

        device = idx.device
        B, T = idx.shape

        # Positional
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # -> [tok_emb]_(B,T,C_e)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C_e)
        x = tok_emb + pos_emb  # [x]_(B,T,C_e)

        # Block
        x = self.transformer_blocks(x)  # [x]_(B,T,C_e)
        x = self.ln_f(x)  # [x]_(B,T,C_e)
        logits = self.lm_head(tok_emb)  # -> [logits]_(B,T,C_v)

        return logits


def gptmm(pretrained=False, **kwargs):
    """GPTmm model"""
    model = GPTmm(**kwargs)
    if pretrained:
        raise NotImplementedError("Pretrained models are not available for GPTmm")
    return model
