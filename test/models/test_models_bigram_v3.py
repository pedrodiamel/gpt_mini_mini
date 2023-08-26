import os
import torch
import torch.nn as nn
from torch.nn import functional as F


PATH_DATASET = "/.datasets/llms/tinyshakespeare"
NAME_DATASET = "input.txt"
PATHNAME_DATASET = os.path.join(PATH_DATASET, NAME_DATASET)

# hyperparameters
batch_size = 64
block_size = 256  # maximun context length for prediction
max_iters = 5000
eval_interval = 300
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 384
n_heads = 6
n_layers = 6
dropout = 0.2
# --------


torch.manual_seed(1337)


#### DATA #####

# read it in to inspect it
# curl https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o $PATHNAME_DATASET
with open(PATHNAME_DATASET, "r", encoding="utf-8") as f:
    text = f.read()


# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
# Examples tools
# https://github.com/google/sentencepiece
# https://github.com/openai/tiktoken
#
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
decode = lambda l: "".join([itos[i] for i in l])  # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
# Let's now split up the data into train and validation sets
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = loss.mean()
    model.train()
    return out


#### MODEL #####


class Head(nn.Module):
    """One head of self-attention"""

    def __init__(self, head_size):
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
    """Multiple heads pf self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.wo = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)  # [x]_(B,T,E)
        x = self.dropout(self.wo(x))  # [x]_(B,T,E)
        return x


class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
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

    def __init__(self, n_embd, n_heads):
        super().__init__()
        head_size = n_embd // n_heads
        self.sa_head = MultiHeadAttention(n_heads, head_size)  # i.e. 4 heads of 8-dimensional self-attention
        self.ff_head = FeedForward(n_embd)  # E \to E
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa_head(self.ln1(x))  # [x]_(B,T,E)
        x = x + self.ff_head(self.ln2(x))  # [x]_(B,T,E)
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # f(x) \in {B,T,C}
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)  # [V,E]
        self.position_embedding_table = nn.Embedding(block_size, n_embd)  # [T,E]
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(n_embd, n_heads) for _ in range(n_layers)]
        )  # E \to E

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)  # E \to V

    def forward(self, idx, targets=None):
        # [idx]_(B,T)
        # [targets]_(B,T)

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

        if targets is None:
            loss = None
        else:
            # B,T,C = logits
            # logits = logits.view(-1, C) or .view(B*T, C)
            # tagets = targets.view(-1) or targets.view(B*T)
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


### TRAIN ###

model = BigramLanguageModel(vocab_size)
m = model.to(device)

opt = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, eval loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = model(xb, yb)

    # evaluate
    logits, loss = m(xb, yb)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
