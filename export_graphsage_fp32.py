
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Amazon
from torch_geometric.nn import SAGEConv


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


checkpoint = torch.load(
    "amazon_photo_graphsage_fp32.pt",
    map_location="cpu",
    weights_only=False,
)

dataset = Amazon(root="data/amazon", name="Photo")
data = dataset[0]

model = GraphSAGE(
    checkpoint["input_features"],
    checkpoint["hidden_channels"],
    checkpoint["num_classes"],
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

with torch.no_grad():
    logits = model(data.x, data.edge_index)

test_idx = torch.tensor(checkpoint["test_idx"], dtype=torch.long)
test_logits = logits[test_idx]
test_labels = data.y[test_idx]
predictions = test_logits.argmax(dim=1)

accuracy = (predictions == test_labels).float().mean().item()

test_logits.numpy().astype("float32").tofile(
    "amazon_photo_test_logits_fp32.bin"
)

test_labels.numpy().astype("int64").tofile(
    "amazon_photo_test_labels_int64.bin"
)

predictions.numpy().astype("int64").tofile(
    "amazon_photo_test_predictions_fp32.bin"
)

print("All logits shape:", tuple(logits.shape))
print("Test logits shape:", tuple(test_logits.shape))
print("Test accuracy:", accuracy)
print("Saved test logits, labels and predictions.")
