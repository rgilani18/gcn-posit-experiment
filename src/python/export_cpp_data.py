import torch
import numpy as np

# Load exported PyTorch data
features = torch.load("cora_features.pt", weights_only=True)
edge_index = torch.load("cora_norm_edge_index.pt", weights_only=True)
edge_weight = torch.load("cora_norm_edge_weight.pt", weights_only=True)

labels = torch.load("cora_labels.pt", weights_only=True)
test_mask = torch.load("cora_test_mask.pt", weights_only=True)

weights = torch.load("gcn_pruned_weights.pt", weights_only=True)

# Convert and save as raw binary files

features.numpy().astype(np.float32).tofile(
    "features_fp32.bin"
)

edge_index.numpy().astype(np.int64).tofile(
    "norm_edge_index_int64.bin"
)

edge_weight.numpy().astype(np.float32).tofile(
    "norm_edge_weight_fp32.bin"
)

weights["conv1_weight"].numpy().astype(np.float32).tofile(
    "conv1_weight_fp32.bin"
)

weights["conv1_bias"].numpy().astype(np.float32).tofile(
    "conv1_bias_fp32.bin"
)

weights["conv2_weight"].numpy().astype(np.float32).tofile(
    "conv2_weight_fp32.bin"
)

weights["conv2_bias"].numpy().astype(np.float32).tofile(
    "conv2_bias_fp32.bin"
)

labels.numpy().astype(np.int64).tofile(
    "labels_int64.bin"
)

test_mask.numpy().astype(np.uint8).tofile(
    "test_mask_uint8.bin"
)

print("C++ data export complete.")
print("Features:", features.shape)
print("Edge index:", edge_index.shape)
print("Edge weights:", edge_weight.shape)
print("Conv1 weight:", weights["conv1_weight"].shape)
print("Conv1 bias:", weights["conv1_bias"].shape)
print("Conv2 weight:", weights["conv2_weight"].shape)
print("Conv2 bias:", weights["conv2_bias"].shape)
print("Labels:", labels.shape)
print("Test mask:", test_mask.shape)
