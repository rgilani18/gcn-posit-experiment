import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.nn.conv.gcn_conv import gcn_norm


dataset = Planetoid(root="data", name="Cora")
data = dataset[0]

# Reproduce PyG GCN normalization exactly:
# A_hat = D^(-1/2) (A + I) D^(-1/2)
edge_index, edge_weight = gcn_norm(
    data.edge_index,
    edge_weight=None,
    num_nodes=data.num_nodes,
    add_self_loops=True,
    dtype=data.x.dtype
)

# Export normalized graph
torch.save(edge_index.cpu(), "cora_norm_edge_index.pt")
torch.save(edge_weight.cpu(), "cora_norm_edge_weight.pt")

print("Normalized graph exported")
print("Normalized edges:", edge_index.shape)
print("Edge weights:", edge_weight.shape)
print("Data type:", edge_weight.dtype)
