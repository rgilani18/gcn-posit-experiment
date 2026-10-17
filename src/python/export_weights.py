import torch
from torch_geometric.datasets import Planetoid

from gcn_model import GCN


dataset = Planetoid(root="data", name="Cora")

model = GCN(
    input_features=dataset.num_features,
    hidden_channels=16,
    num_classes=dataset.num_classes
)

model.load_state_dict(
    torch.load("gcn_cora_pruned50_fp32.pth", map_location="cpu")
)

model.eval()


# Export GCN weights and biases
torch.save(
    {
        "conv1_weight": model.conv1.lin.weight.detach().cpu(),
        "conv1_bias": model.conv1.bias.detach().cpu(),
        "conv2_weight": model.conv2.lin.weight.detach().cpu(),
        "conv2_bias": model.conv2.bias.detach().cpu(),
    },
    "gcn_pruned_weights.pt"
)

print("Exported: gcn_pruned_weights.pt")
print("conv1 weight:", model.conv1.lin.weight.shape)
print("conv1 bias:", model.conv1.bias.shape)
print("conv2 weight:", model.conv2.lin.weight.shape)
print("conv2 bias:", model.conv2.bias.shape)
