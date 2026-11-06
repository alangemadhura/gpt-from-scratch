"""
Character-level GPT (decoder-only Transformer), following Andrej Karpathy's
"Let's build GPT: from scratch, in code, spelled out".

Run:  python gpt.py
Needs: torch, and input.txt (Tiny Shakespeare) in the same folder.
Get the data:
  wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------------- hyperparameters ----------------
# Small settings so it runs on a CPU/laptop. For a GPU, use the "big" values.
batch_size = 32        # big: 64   independent sequences processed in parallel
block_size = 64        # big: 256  max context length (T)
max_iters = 3000       # big: 5000
eval_interval = 300    # how often to print train/val loss
learning_rate = 1e-3   # big: 3e-4
eval_iters = 100
n_embd = 128           # big: 384  embedding dimension (C)
n_head = 4             # big: 6    attention heads
n_layer = 4            # big: 6    number of Transformer blocks
dropout = 0.1          # big: 0.2
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
torch.manual_seed(1337)
print("Using device:", device)

# ---------------- data ----------------
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Character-level "tokenizer": every unique character is a token.
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join(itos[i] for i in l)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))               # 90% train / 10% validation
train_data, val_data = data[:n], data[n:]


def get_batch(split):
    """Random chunks of text. y is x shifted by one (next-character targets)."""
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# ---------------- model ----------------
class Head(nn.Module):
    """One head of causal (masked) self-attention."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # lower-triangular matrix: a token may only look at earlier tokens
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)      # (B, T, hs)  "what do I contain?"
        q = self.query(x)    # (B, T, hs)  "what am I looking for?"
        # affinities, scaled by sqrt(head_size) so softmax doesn't get too peaky
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5   # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # hide the future
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)    # (B, T, hs)  "what I will communicate"
        return wei @ v       # (B, T, hs)


class MultiHeadAttention(nn.Module):
    """Several attention heads in parallel, concatenated, then projected."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """Per-token MLP: the 'thinking' step after tokens have communicated."""

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


class Block(nn.Module):
    """Transformer block: communication (attention) then computation (MLP),
    with residual connections and pre-LayerNorm."""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))     # residual connection
        x = x + self.ffwd(self.ln2(x))   # residual connection
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)                               # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                                                # (B, T, vocab)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]              # crop to context window
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]                    # last time step only
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # sample
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ---------------- training ----------------
model = GPTLanguageModel().to(device)
print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for it in range(max_iters):
    if it % eval_interval == 0 or it == max_iters - 1:
        losses = estimate_loss()
        print(f"step {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# ---------------- generate ----------------
context = torch.zeros((1, 1), dtype=torch.long, device=device)  # start from token 0 ("\n")
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))

torch.save(model.state_dict(), 'gpt_shakespeare.pt')
# To write more text to a file:
# open('more.txt', 'w').write(decode(model.generate(context, max_new_tokens=10000)[0].tolist()))