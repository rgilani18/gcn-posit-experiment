import torch
from torch_geometric.datasets import Planetoid

from gcn_model import GCN


# Load Cora
dataset = Planetoid(root="data", name="Cora")
data = dataset[0]

# Create model
model = GCN(
    input_features=dataset.num_features,
    hidden_channels=16,
    num_classes=dataset.num_classes
)

# Load 50% pruned model
model.load_state_dict(
    torch.load("gcn_cora_pruned50_fp32.pth", map_location="cpu")
)

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


print("50% Pruned FP32 Results")
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

