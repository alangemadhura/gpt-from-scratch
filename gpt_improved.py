"""
Character-level GPT with two improvements over the tutorial version:
  1. Best-checkpoint saving: weights are saved only when validation loss
     improves, so overfitting at the end of training can't hurt you.
  2. Loss logging: train/val losses are written to loss_log.csv and, if
     matplotlib is installed, plotted to loss_curve.png.

Run:  python gpt_improved.py
Needs: torch, input.txt (Tiny Shakespeare). Optional: matplotlib.
Saves: gpt_shakespeare.pt (BEST weights, works with generate.py)
"""
import csv
import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------------- hyperparameters ----------------
batch_size = 32
block_size = 64
max_iters = 3000
eval_interval = 300
eval_iters = 100
learning_rate = 1e-3
n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.1
patience = 3           # stop early after this many evals with no val improvement
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
torch.manual_seed(1337)
print("Using device:", device)

# ---------------- data ----------------
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join(itos[i] for i in l)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]


def get_batch(split):
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
        out[split] = losses.mean().item()
    model.train()
    return out


# ---------------- model ----------------
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ self.value(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, n_embd // n_head)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
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
        x = self.token_embedding_table(idx) + self.position_embedding_table(torch.arange(T, device=device))
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
        return idx


# ---------------- training with best-checkpoint + early stopping ----------------
model = GPTLanguageModel().to(device)
print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

best_val = float('inf')
bad_evals = 0
history = []   # (step, train_loss, val_loss)

for it in range(max_iters):
    if it % eval_interval == 0 or it == max_iters - 1:
        losses = estimate_loss()
        history.append((it, losses['train'], losses['val']))
        print(f"step {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        if losses['val'] < best_val:
            best_val = losses['val']
            bad_evals = 0
            torch.save(model.state_dict(), 'gpt_shakespeare.pt')
            print(f"  -> new best val loss {best_val:.4f}, checkpoint saved")
        else:
            bad_evals += 1
            if bad_evals >= patience:
                print(f"Early stopping: no val improvement for {patience} evaluations.")
                break

    xb, yb = get_batch('train')
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(f"\nBest validation loss: {best_val:.4f}")

# ---------------- save log + plot ----------------
with open('loss_log.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['step', 'train_loss', 'val_loss'])
    w.writerows(history)
print("Saved loss_log.csv")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    steps = [h[0] for h in history]
    plt.figure(figsize=(7, 4))
    plt.plot(steps, [h[1] for h in history], label='train')
    plt.plot(steps, [h[2] for h in history], label='val')
    plt.xlabel('step'); plt.ylabel('loss'); plt.legend(); plt.title('Training curve')
    plt.tight_layout()
    plt.savefig('loss_curve.png', dpi=150)
    print("Saved loss_curve.png")
except ImportError:
    print("matplotlib not installed; skipped plot (pip install matplotlib to enable).")

# ---------------- sample from the BEST weights ----------------
model.load_state_dict(torch.load('gpt_shakespeare.pt', map_location=device))
model.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))