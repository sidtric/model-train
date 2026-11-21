import torch
import torch.nn as nn
from torch.nn import functional as F

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

print(len(text))

if "*** START OF" in text:
    start = text.find("*** START OF")
    start = text.find("\n", start) + 1
    end = text.find("*** END OF")
    text = text[start:end]
text = text.strip()

print(len(text))

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(vocab_size)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
print(len(train_data), len(val_data))

torch.manual_seed(1337)
block_size = 8
batch_size = 4

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')
print(xb.shape)
# print(decode(xb[0].tolist()))
# print(decode(yb[0].tolist()))

block_size = 64
batch_size = 32
n_embd = 128
n_head = 4
n_layer = 4

xb, yb = get_batch('train')
print(xb.shape)

token_embedding_table = nn.Embedding(vocab_size, n_embd)
position_embedding_table = nn.Embedding(block_size, n_embd)

tok_emb = token_embedding_table(xb)
pos_emb = position_embedding_table(torch.arange(block_size))
x = tok_emb + pos_emb
print(x.shape)

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        return wei @ v

head = Head(32)
out = head(x)
# print(out.shape)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)

mha = MultiHeadAttention(n_head, n_embd // n_head)
out = mha(x)
# print(out.shape)

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)

ffwd = FeedForward(n_embd)
out = ffwd(out)
# print(out.shape)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

block = Block(n_embd, n_head)
out = block(x)
# print(out.shape)

class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

model = TinyTransformer()
logits = model(xb)
print(logits.shape)
print(sum(p.numel() for p in model.parameters()), "parameters")

B, T, C = logits.shape
loss = F.cross_entropy(logits.view(B*T, C), yb.view(B*T))
print(loss.item())

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for step in range(2000):
    xb, yb = get_batch('train')
    logits = model(xb)
    B, T, C = logits.shape
    loss = F.cross_entropy(logits.view(B*T, C), yb.view(B*T))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print(f"step {step}: loss {loss.item():.3f}")

print("final loss:", loss.item())

def generate(idx, max_new_tokens):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

context = torch.tensor([encode("\n")])
with torch.no_grad():
    out = generate(context, 500)
print(decode(out[0].tolist()))

torch.save({
    'model': model.state_dict(),
    'chars': chars,
}, 'pretrained.pt')
print("saved pretrained.pt")