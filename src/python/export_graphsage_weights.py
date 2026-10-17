
import torch

checkpoint = torch.load(
    "amazon_photo_graphsage_fp32.pt",
    map_location="cpu",
    weights_only=False,
)

state = checkpoint["model_state_dict"]

for name, tensor in state.items():
    filename = name.replace(".", "_") + "_fp32.bin"
    tensor.detach().cpu().contiguous().numpy().astype(
        "float32"
    ).tofile(filename)
    print(f"{filename}: shape={tuple(tensor.shape)}")

print("All GraphSAGE parameters exported.")
