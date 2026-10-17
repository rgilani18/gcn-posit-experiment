import torch
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root="data", name="Cora")
data = dataset[0]

# Export node features
torch.save(data.x.cpu(), "cora_features.pt")

# Export graph edges
torch.save(data.edge_index.cpu(), "cora_edge_index.pt")

# Export labels and masks
torch.save(data.y.cpu(), "cora_labels.pt")
torch.save(data.test_mask.cpu(), "cora_test_mask.pt")

print("Exported Cora data")
print("Features:", data.x.shape)
print("Edges:", data.edge_index.shape)
print("Labels:", data.y.shape)
print("Test mask:", data.test_mask.shape)
