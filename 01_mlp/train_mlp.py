import torch
import torch.nn as nn

X = torch.tensor([[0.,0.], [0.,1.], [1.,0.], [1.,1.]])
y = torch.tensor([[0.], [1.], [1.], [0.]])

class TinyMLP(nn.Module):
    def __init__(self, hidden_size=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.net(x) 

model = TinyMLP()

loss_fn = nn.BCEWithLogitsLoss()   
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

with torch.no_grad():
    print("Before training:", torch.sigmoid(model(X)).squeeze())

for step in range(500):
    optimizer.zero_grad()
    predictions = model(X)
    loss = loss_fn(predictions, y)
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"step {step}: loss = {loss.item():.4f}")

with torch.no_grad():
    print("After training:", torch.sigmoid(model(X)).squeeze())
    print("Target:        ", y.squeeze())