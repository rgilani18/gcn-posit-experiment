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

# Load trained FP32 weights
model.load_state_dict(torch.load("gcn_cora_fp32.pth", map_location="cpu"))

model.eval()


# Collect all trainable weights
weights = []

for name, param in model.named_parameters():
    if "weight" in name:
        weights.append(param.detach().abs().flatten())

all_weights = torch.cat(weights)

# Find 50th percentile threshold
threshold = torch.quantile(all_weights, 0.50)

print(f"Pruning threshold: {threshold.item():.8f}")


# Apply 50% magnitude pruning
total_weights = 0
zero_weights = 0

with torch.no_grad():
    for name, param in model.named_parameters():
        if "weight" in name:
            mask = param.abs() <= threshold
            param[mask] = 0

            total_weights += param.numel()
            zero_weights += (param == 0).sum().item()


sparsity = zero_weights / total_weights

print(f"Total weights: {total_weights}")
print(f"Zero weights: {zero_weights}")
print(f"Sparsity: {sparsity * 100:.2f}%")


# Save pruned model
torch.save(model.state_dict(), "gcn_cora_pruned50_fp32.pth")

print("Saved: gcn_cora_pruned50_fp32.pth")
