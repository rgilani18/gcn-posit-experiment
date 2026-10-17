import torch
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root="data", name="Cora")
data = dataset[0]

print("=== Loaded Planetoid Cora ===")
print("Nodes:", data.num_nodes)
print("Features:", tuple(data.x.shape))
print("Classes:", dataset.num_classes)
print("Original edges:", data.edge_index.shape[1])
print("Train mask:", int(data.train_mask.sum()))
print("Validation mask:", int(data.val_mask.sum()))
print("Test mask:", int(data.test_mask.sum()))

print("\n=== Exported tensors ===")
for name in [
    "cora_features.pt",
    "cora_edge_index.pt",
    "cora_labels.pt",
    "cora_test_mask.pt",
    "cora_norm_edge_index.pt",
    "cora_norm_edge_weight.pt",
]:
    obj = torch.load(name, map_location="cpu", weights_only=True)
    print(f"{name}: shape={tuple(obj.shape)}, dtype={obj.dtype}")

print("\n=== Mask consistency ===")
saved_test = torch.load(
    "cora_test_mask.pt", map_location="cpu", weights_only=True
).bool()
print("Saved test mask matches Planetoid:", torch.equal(saved_test, data.test_mask))

print("\n=== Labels and features ===")
saved_labels = torch.load(
    "cora_labels.pt", map_location="cpu", weights_only=True
)
saved_features = torch.load(
    "cora_features.pt", map_location="cpu", weights_only=True
)
print("Labels match:", torch.equal(saved_labels, data.y))
print("Features match:", torch.equal(saved_features, data.x))
