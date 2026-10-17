
import torch
from torch_geometric.datasets import Amazon

checkpoint = torch.load(
    "amazon_photo_graphsage_fp32.pt",
    map_location="cpu",
    weights_only=False,
)

dataset = Amazon(root="data/amazon", name="Photo")
data = dataset[0]

test_idx = torch.tensor(checkpoint["test_idx"], dtype=torch.long)

data.x.contiguous().numpy().astype("float32").tofile(
    "amazon_photo_features_fp32.bin"
)

data.edge_index[0].contiguous().numpy().astype("int64").tofile(
    "amazon_photo_edge_src_int64.bin"
)

data.edge_index[1].contiguous().numpy().astype("int64").tofile(
    "amazon_photo_edge_dst_int64.bin"
)

data.y[test_idx].contiguous().numpy().astype("int64").tofile(
    "amazon_photo_test_labels_int64.bin"
)

test_idx.numpy().astype("int64").tofile(
    "amazon_photo_test_indices_int64.bin"
)

print("Features:", tuple(data.x.shape))
print("Edges:", tuple(data.edge_index.shape))
print("Test nodes:", len(test_idx))
print("Graph data exported successfully.")

