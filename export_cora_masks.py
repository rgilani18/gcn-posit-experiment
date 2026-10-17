import numpy as np
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root="data", name="Cora")
data = dataset[0]

masks = {
    "train_mask_uint8.bin": data.train_mask,
    "val_mask_uint8.bin": data.val_mask,
    "test_mask_uint8.bin": data.test_mask,
}

for filename, mask in masks.items():
    values = mask.cpu().numpy().astype(np.uint8)
    values.tofile(filename)
    print(f"{filename}: {int(values.sum())} selected nodes")

print("Official Planetoid masks exported.")
