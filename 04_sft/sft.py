import torch
import torch.nn as nn
from torch.nn import functional as F

checkpoint = torch.load("../03_pretrain/pretrained.pt")
chars = checkpoint['chars']
print(len(chars))
print(list(checkpoint['model'].keys())[:5])

vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

block_size = 64
batch_size = 32
n_embd = 128
n_head = 4
n_layer = 4

print(decode(encode("What is a habit?")))

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

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)

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
        return self.lm_head(x)

model = TinyTransformer()
model.load_state_dict(checkpoint['model'])
print("pretrained model loaded")

def generate(idx, max_new_tokens):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

def ask(question):
    prompt = f"Q: {question}\nA:"
    idx = torch.tensor([encode(prompt)])
    with torch.no_grad():
        out = generate(idx, 150)
    return decode(out[0].tolist())

print("--- BEFORE SFT---")
print(ask("What is a habit?"))

qa_pairs = [
    ("What is a habit?", "A habit is a behavior repeated until it is automatic."),
    ("Why do small habits matter?", "Small habits compound over time."),
    ("What is 1 percent better?", "Getting a little better every day."),
    ("What is the first law?", "Make it obvious."),
    ("What is the second law?", "Make it attractive."),
    ("What is the third law?", "Make it easy."),
    ("What is the fourth law?", "Make it satisfying."),
    ("How do I break a bad habit?", "Make it invisible, unattractive, hard."),
    ("What are the four steps?", "Cue, craving, response, reward."),
    ("What is a cue?", "A cue is the trigger for a habit."),
    ("What is a reward?", "A reward is what makes a habit last."),
    ("What is habit stacking?", "Link a new habit to a current one."),
    ("What matters more, goals or systems?", "Systems matter more than goals."),
    ("What is identity based habit?", "Focus on who you want to become."),
    ("What is the two minute rule?", "Start a habit in under two minutes."),
    ("How do I stay consistent?", "Never miss twice."),
    ("What is environment design?", "Shape your space to guide habits."),
    ("How long does a habit take?", "It takes repetition, not time."),
    ("Why do people fail at habits?", "They focus on goals, not systems."),
    ("What is motivation?", "Motivation is overrated, systems win."),
]


sft_text = ""
for q, a in qa_pairs:
    sft_text += f"Q: {q}\nA: {a}\n\n" 

print(len(qa_pairs), "pairs")
print(len(sft_text), "characters")
print(sft_text[:200])

sft_data = torch.tensor(encode(sft_text), dtype=torch.long)
print(sft_data.shape)

def get_batch():
    ix = torch.randint(len(sft_data) - block_size, (batch_size,))
    x = torch.stack([sft_data[i:i+block_size] for i in ix])
    y = torch.stack([sft_data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch()
print(xb.shape)
print(decode(xb[0].tolist()))

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for step in range(500):
    xb, yb = get_batch()
    logits = model(xb)
    B, T, C = logits.shape
    loss = F.cross_entropy(logits.view(B*T, C), yb.view(B*T))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"step {step}: loss {loss.item():.3f}")

print("final loss:", loss.item())

def ask_clean(question):
    out = ask(question)
    answer = out.split("\nA:")[1].split("\n\n")[0]
    return answer.strip()

print("--- AFTER SFT---")
print("Q: What is a habit?")
print("A:", ask_clean("What is a habit?"))
print()
print("Q: What is the third law?")
print("A:", ask_clean("What is the third law?"))
print()
print("Q: How do I build a reading habit?")
print("A:", ask_clean("How do I build a reading habit?"))

torch.save({'model': model.state_dict(), 'chars': chars}, 'sft.pt')
