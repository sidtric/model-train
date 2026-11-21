import torch
import torch.nn as nn
from torch.nn import functional as F

checkpoint = torch.load("../05_dpo/dpo.pt")
chars = checkpoint['chars']
print(len(chars))

vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

block_size = 64
n_embd = 128
n_head = 4
n_layer = 4

print(decode(encode("Make it easy.")))

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

print("classes ready")
    
policy = TinyTransformer()
policy.load_state_dict(checkpoint['model'])

reference = TinyTransformer()
reference.load_state_dict(checkpoint['model'])
reference.eval()
for p in reference.parameters():
    p.requires_grad = False

print("policy and reference loaded")

def answer_logprob(model, question, answer):
    prompt = encode(f"Q: {question}\nA: ")
    full = prompt + encode(answer + "\n\n")
    n_answer = len(full) - len(prompt)

    full = full[-(block_size + 1):]
    x = torch.tensor([full[:-1]])
    y = torch.tensor([full[1:]])

    logits = model(x)
    logprobs = F.log_softmax(logits, dim=-1)
    token_logprobs = logprobs.gather(2, y.unsqueeze(-1)).squeeze(-1)
    return token_logprobs[0, -n_answer:].sum()

def generate(model, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

def ask(model, question):
    prompt = f"Q: {question}\nA:"
    idx = torch.tensor([encode(prompt)])
    with torch.no_grad():
        out = decode(generate(model, idx, 80)[0].tolist())
    return out.split("\nA:")[1].split("\n\n")[0].strip()

tasks = [
    ("What is the first law?", "obvious"),
    ("What is the second law?", "attractive"),
    ("What is the third law?", "easy"),
    ("What is the fourth law?", "satisfying"),
    ("How do I stay consistent?", "twice"),
]

def reward(answer, keyword):
    if keyword in answer.lower():
        return 1.0
    return 0.0

print(reward("Make it easy.", "easy"))
print(reward("Make it attractive.", "easy"))

group_size = 8

q, keyword = tasks[2]
answers = [ask(policy, q) for _ in range(group_size)]
rewards = [reward(a, keyword) for a in answers]

print("Q:", q)
for a, r in zip(answers, rewards):
    print(r, "|", a)

r = torch.tensor(rewards)
advantages = (r - r.mean()) / (r.std() + 1e-6)

print("mean reward:", r.mean().item())
for a, rw, adv in zip(answers, rewards, advantages):
    print(f"{rw} | adv {adv.item():+.2f} | {a}")

kl_coef = 0.1
optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-5)

for step in range(20):
    step_rewards = []
    for q, keyword in tasks:
        answers = [ask(policy, q) for _ in range(group_size)]
        rewards = torch.tensor([reward(a, keyword) for a in answers])
        step_rewards.append(rewards.mean().item())

        if rewards.std() == 0:
            continue
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-6)

        loss = 0
        for a, adv in zip(answers, advantages):
            pol = answer_logprob(policy, q, a)
            with torch.no_grad():
                ref = answer_logprob(reference, q, a)
            loss = loss - adv * pol + kl_coef * 0.5 * (pol - ref) ** 2
        loss = loss / group_size

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg = sum(step_rewards) / len(step_rewards)
    print(f"step {step}: avg reward {avg:.2f}")

def accuracy(model, q, keyword, n=20):
    hits = sum(reward(ask(model, q), keyword) for _ in range(n))
    return hits / n

print("\n--- before (reference) vs after (policy) ---")
for q, keyword in tasks:
    before = accuracy(reference, q, keyword)
    after = accuracy(policy, q, keyword)
    print(f"{q:<28} before {before:.0%}   after {after:.0%}   | {ask(policy, q)}")

torch.save({'model': policy.state_dict(), 'chars': chars}, 'grpo.pt')
   