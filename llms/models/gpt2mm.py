"""PyTorch NanoGPT base in OpenAI GPT-2 model"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F


__all__ = ["GPT2mm", "gpt2mm"]


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size, dropout=True, bias=False):
        r"""Multi-Head Attention module with Flash Attention support
        Args:
            num_head: number of heads
            num_embd: embedding dimensionality
            block_size: maximum sequence length
            dropout: dropout probability
            bias: whether to use bias
        """
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout

        # number of heads
        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.c_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
            )

    def forward(self, x):
        B, T, C = x.size()  # batch, time, channels

        # [x]_(B,T,H)
        # x = c_attN(x) -> [x]_(B,T,3H)
        # q,k,v = split(x, H, dim=2) -> [q,k,v]_(B,T,H)
        q, k, v = self.c_attn(x).split(self.embed_dim, dim=2)

        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, NH, T, H)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, NH, T, H)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, NH, T, H)

        # causal self-attention; Self-attend: (B, NH, T, H) x (B, NH, H, T) -> (B, NH, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, NH, T, T) x (B, NH, T, H) -> (B, NH, T, H)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class FeedForward(nn.Module):
    def __init__(self, embed_dim, dropout=True, bias=False):
        r"""A simple linear layer followed by a non-linearity"""
        super().__init__()
        self.c_fc = nn.Linear(embed_dim, 4 * embed_dim, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size, dropout=True, bias=False):
        r"""A transformer block"""
        super().__init__()
        self.ln_1 = LayerNorm(embed_dim, bias=bias)
        self.attn = MultiHeadAttention(embed_dim, num_heads, block_size, dropout, bias)
        self.ln_2 = LayerNorm(embed_dim, bias=bias)
        self.mlp = FeedForward(embed_dim, dropout, bias)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))  # [x]_(B,T,E)
        x = x + self.mlp(self.ln_2(x))  # [x]_(B,T,E)
        return x


class GPT2mm(nn.Module):
    def __init__(self, n_layers, vocab_size, n_embd, n_heads, block_size, dropout=0.2, bias=False, predition=False):
        super().__init__()
        assert vocab_size > 0, "vocab_size must be specified and >0"
        assert block_size > 0, "block_size must be specified and >0"
        assert n_embd > 0, "n_embd must be specified and >0"
        assert n_heads > 0, "n_head must be specified and >0"

        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_heads = n_heads
        self.predition = predition

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(vocab_size, n_embd),
                wpe=nn.Embedding(block_size, n_embd),
                drop=nn.Dropout(dropout),
                h=nn.ModuleList(
                    [
                        TransformerBlock(
                            n_embd,
                            n_heads,
                            block_size,
                            dropout,
                            bias,
                        )
                        for _ in range(n_layers)
                    ]
                ),
                ln_f=LayerNorm(n_embd, bias=bias),
            )
        )

        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        # init all weights
        self.apply(self._init_weights)

        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.shape  # batch size (B) and length of sequence (T)

        assert T <= self.block_size, f"Cannot forward sequence of length {T}, block size is only {self.block_size}"

        # Learned positional embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)  # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (B, T, E)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (1, T, E)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if not self.predition:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim

        return logits

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]


def gpt2mm(pretrained=False, **kwargs):
    """GPT2mm model"""
    model = GPT2mm(**kwargs)
    if pretrained:
        raise NotImplementedError("Pretrained models are not available for GPTmm")
    return model
