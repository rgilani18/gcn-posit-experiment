
import copy
import random

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Amazon
from torch_geometric.nn import SAGEConv
from sklearn.model_selection import train_test_split


# Reproducibility
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cpu")
print("Device:", device)

# Load Amazon Photo
dataset = Amazon(root="data/amazon", name="Photo")
data = dataset[0].to(device)

num_nodes = data.num_nodes
num_classes = dataset.num_classes

# Fixed stratified 60/20/20 split
indices = list(range(num_nodes))
labels = data.y.cpu().numpy()

train_idx, temp_idx = train_test_split(
    indices,
    test_size=0.40,
    random_state=SEED,
    stratify=labels,
)

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.50,
    random_state=SEED,
    stratify=labels[temp_idx],
)

train_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)

train_mask[train_idx] = True
val_mask[val_idx] = True
test_mask[test_idx] = True

train_mask = train_mask.to(device)
val_mask = val_mask.to(device)
test_mask = test_mask.to(device)


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


model = GraphSAGE(
    in_channels=dataset.num_node_features,
    hidden_channels=64,
    num_classes=num_classes,
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01,
    weight_decay=5e-4,
)

best_val_acc = -1.0
best_state = None
best_epoch = 0
patience = 30
epochs_without_improvement = 0
MAX_EPOCHS = 200

print("Nodes:", num_nodes)
print("Edges:", data.num_edges)
print("Features:", dataset.num_node_features)
print("Classes:", num_classes)
print("Split sizes:", int(train_mask.sum()), int(val_mask.sum()),
      int(test_mask.sum()))
print("Parameters:", sum(p.numel() for p in model.parameters()))

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    optimizer.zero_grad()

    logits = model(data.x, data.edge_index)
    loss = F.cross_entropy(logits[train_mask], data.y[train_mask])

    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        pred = logits.argmax(dim=1)

        train_acc = (pred[train_mask] == data.y[train_mask]).float().mean().item()
        val_acc = (pred[val_mask] == data.y[val_mask]).float().mean().item()

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch
        best_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epoch == 1 or epoch % 10 == 0:
        print(
            f"Epoch {epoch:03d} | Loss {loss.item():.4f} "
            f"| Train {train_acc:.4f} | Val {val_acc:.4f}"
        )

    if epochs_without_improvement >= patience:
        print("Early stopping.")
        break

# Restore the best validation checkpoint
model.load_state_dict(best_state)
model.eval()

with torch.no_grad():
    logits = model(data.x, data.edge_index)
    pred = logits.argmax(dim=1)
    test_acc = (pred[test_mask] == data.y[test_mask]).float().mean().item()

torch.save(
    {
        "model_state_dict": model.cpu().state_dict(),
        "input_features": dataset.num_node_features,
        "hidden_channels": 64,
        "num_classes": num_classes,
        "seed": SEED,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    },
    "amazon_photo_graphsage_fp32.pt",
)

print("\nBest epoch:", best_epoch)
print(f"Best validation accuracy: {best_val_acc:.4f}")
print(f"Test accuracy: {test_acc:.4f}")
print("Saved: amazon_photo_graphsage_fp32.pt")
