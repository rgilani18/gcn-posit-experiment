import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid

from gcn_model import GCN


# Load Cora
dataset = Planetoid(root="data", name="Cora")
data = dataset[0]

# CPU only
device = torch.device("cpu")

model = GCN(
    input_features=dataset.num_features,
    hidden_channels=16,
    num_classes=dataset.num_classes
).to(device)

data = data.to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01,
    weight_decay=5e-4
)


def evaluate():
    model.eval()

    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        pred = logits.argmax(dim=1)

        train_acc = (
            pred[data.train_mask] == data.y[data.train_mask]
        ).float().mean().item()

        val_acc = (
            pred[data.val_mask] == data.y[data.val_mask]
        ).float().mean().item()

        test_acc = (
            pred[data.test_mask] == data.y[data.test_mask]
        ).float().mean().item()

    return train_acc, val_acc, test_acc


# Training
for epoch in range(1, 201):
    model.train()

    optimizer.zero_grad()

    logits = model(data.x, data.edge_index)

    loss = F.cross_entropy(
        logits[data.train_mask],
        data.y[data.train_mask]
    )

    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        train_acc, val_acc, test_acc = evaluate()

        print(
            f"Epoch {epoch:03d} | "
            f"Loss {loss.item():.4f} | "
            f"Train {train_acc:.4f} | "
            f"Val {val_acc:.4f} | "
            f"Test {test_acc:.4f}"
        )


# Final evaluation
train_acc, val_acc, test_acc = evaluate()

print("\nFinal Results")
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")


# Save trained FP32 model
torch.save(model.state_dict(), "gcn_cora_fp32.pth")

print("\nSaved: gcn_cora_fp32.pth")
