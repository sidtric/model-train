import torch
import torch.nn as nn
from torch.nn import functional as F

checkpoint = torch.load("../04_sft/sft.pt")
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

print(decode(encode("Never miss twice.")))

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

pairs = [
    ("How do I stay consistent?", "Never miss twice.", "Use more willpower."),
    ("What matters more, goals or systems?", "Systems matter more than goals.", "Goals matter most."),
    ("What is motivation?", "Motivation is overrated, systems win.", "Wait until you feel ready."),
    ("How do I break a bad habit?", "Make it invisible, unattractive, hard.", "Just stop doing it."),
    ("What is the third law?", "Make it easy.", "Make it attractive."),
    ("How long does a habit take?", "It takes repetition, not time.", "Exactly 21 days."),
    ("Why do small habits matter?", "Small habits compound over time.", "Small habits do not matter."),
    ("What is the two minute rule?", "Start a habit in under two minutes.", "Work for two hours a day."),
    ("What is identity based habit?", "Focus on who you want to become.", "Focus only on results."),
    ("What is environment design?", "Shape your space to guide habits.", "Ignore your surroundings."),
]

for q, good, bad in pairs:
    p = f"Q: {q}\nA: "
    print(len(p + good + "\n\n"), len(p + bad + "\n\n"))

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

q, good, bad = pairs[0]
print(q)
print("chosen  :", answer_logprob(policy, q, good).item())
print("rejected:", answer_logprob(policy, q, bad).item())    

beta = 0.5 

def dpo_loss(question, chosen, rejected):
    pol_chosen = answer_logprob(policy, question, chosen)
    pol_rejected = answer_logprob(policy, question, rejected)
    with torch.no_grad():
        ref_chosen = answer_logprob(reference, question, chosen)
        ref_rejected = answer_logprob(reference, question, rejected)

    chosen_gain = pol_chosen - ref_chosen
    rejected_gain = pol_rejected - ref_rejected
    return -F.logsigmoid(beta * (chosen_gain - rejected_gain))

q, good, bad = pairs[0]
print("dpo loss:", dpo_loss(q, good, bad).item())

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

test_questions = ["What is the third law?", "How long does a habit take?", "How do I stay consistent?"]

print("--- BEFORE DPO ---")
for q in test_questions:
    print("Q:", q)
    print("A:", ask(policy, q))

optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-5)

for epoch in range(10):
    total = 0
    for q, good, bad in pairs:
        loss = dpo_loss(q, good, bad)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    if epoch % 5 == 0:
        print(f"epoch {epoch}: avg dpo loss {total / len(pairs):.3f}")

print("final avg loss:", total / len(pairs))

print("--- AFTER DPO ---")
for q in test_questions:
    print("Q:", q)
    print("A:", ask(policy, q))

print("--- scores for 'third law' ---")
q, good, bad = pairs[4]
with torch.no_grad():
    print("chosen   (Make it easy.)       ref:", round(answer_logprob(reference, q, good).item(), 2),
          " policy:", round(answer_logprob(policy, q, good).item(), 2))
    print("rejected (Make it attractive.) ref:", round(answer_logprob(reference, q, bad).item(), 2),
          " policy:", round(answer_logprob(policy, q, bad).item(), 2))

torch.save({'model': policy.state_dict(), 'chars': chars}, 'dpo.pt')
   