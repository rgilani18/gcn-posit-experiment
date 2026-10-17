import torch
from torch_geometric.datasets import Planetoid
from gcn_model import GCN
import numpy as np

dataset = Planetoid(root="data", name="Cora")
data = dataset[0]

model = GCN(
    input_features=dataset.num_node_features,
    hidden_channels=16,
    num_classes=dataset.num_classes
)

model.load_state_dict(torch.load(
    "gcn_cora_pruned50_fp32.pth",
    map_location="cpu"
))

model.eval()

with torch.no_grad():
    logits = model(data.x, data.edge_index)

# Save PyTorch FP32 logits
logits_np = logits.cpu().numpy().astype(np.float32)
logits_np.tofile("golden_logits_fp32.bin")

print("Golden logits shape:", logits.shape)
print("Golden logits dtype:", logits.dtype)
print("Saved: golden_logits_fp32.bin")
