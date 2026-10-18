import os
import sys
import pickle
import numpy as np
import scipy.sparse as sp

def parse_index_file(filename):
    """Parse integer indices from raw text file."""
    index = []
    with open(filename, 'r') as f:
        for line in f:
            index.append(int(line.strip()))
    return index

def build_cora_from_scratch(raw_dir="data/Cora/raw", out_dir="data"):
    print("=============================")
    print("  BUILDING CORA GCN DATASET  ")
    print("=============================")
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph']
    objects = []
    for name in names:
        path = os.path.join(raw_dir, f"ind.cora.{name}")
        with open(path, 'rb') as f:
            objects.append(pickle.load(f, encoding='latin1'))

    x, tx, allx, y, ty, ally, graph = objects
    test_idx_reorder = parse_index_file(os.path.join(raw_dir, "ind.cora.test.index"))
    test_idx_range = np.sort(test_idx_reorder)

    # 1. Reconstruct feature matrix X
    # Following the canonical Kipf & Welling semi-supervised dataset loader
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = features.toarray().astype(np.float32)

    # Row-normalize features: x = x / sum(x)
    row_sums = features.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    features = features / row_sums

    N, F = features.shape
    print(f"[1] Node Features Matrix (X): {N} nodes x {F} word features")

    # 2. Reconstruct Labels Y
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = np.argmax(labels, axis=1).astype(np.int64)
    C = len(np.unique(labels))
    print(f"[2] Labels (Y): {len(labels)} nodes across {C} target classes")

    # 3. Construct Adjacency Matrix A and add self-loops (A + I)
    adj = sp.dok_matrix((N, N), dtype=np.float32)
    for src, neighbors in graph.items():
        for dst in neighbors:
            adj[src, dst] = 1.0
            adj[dst, src] = 1.0  # Undirected

    adj = adj.tocoo()
    print(f"[3] Raw Graph Adjacency: {adj.nnz} directed edges before self-loops")

    # Add identity matrix (self-loops: A_tilde = A + I)
    adj_tilde = adj + sp.eye(N, dtype=np.float32)
    adj_tilde = adj_tilde.tocoo()
    print(f"    Graph with self-loops: {adj_tilde.nnz} directed edges")

    # 4. Compute Symmetric Degree Normalization from Scratch:
    #    D_tilde(i, i) = sum_j (A_tilde(i, j))
    #    A_hat(i, j)   = A_tilde(i, j) / sqrt(D_tilde(i) * D_tilde(j))
    row_sum = np.array(adj_tilde.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(row_sum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0

    src_nodes = adj_tilde.row.astype(np.int64)
    dst_nodes = adj_tilde.col.astype(np.int64)
    norm_edge_weights = (d_inv_sqrt[src_nodes] * d_inv_sqrt[dst_nodes]).astype(np.float32)

    E = len(src_nodes)
    print(f"[4] Symmetric Normalization: {E} normalized graph convolution edges computed")

    # Combine edge index into [src; dst] continuous format (2 * E)
    edge_index = np.concatenate([src_nodes, dst_nodes]).astype(np.int64)

    # 5. Build Standard Planetoid Masks (140 train, 500 val, 1000 test)
    train_mask = np.zeros(N, dtype=np.uint8)
    val_mask = np.zeros(N, dtype=np.uint8)
    test_mask = np.zeros(N, dtype=np.uint8)

    train_mask[:140] = 1
    val_mask[200:500] = 1
    test_mask[test_idx_range] = 1

    print(f"[5] Split Masks: Train={train_mask.sum()}, Val={val_mask.sum()}, Test={test_mask.sum()}")

    # 6. Export Binary Files to data/
    print("\n[6] Exporting clean raw binary files to data/:")
    features.tofile(os.path.join(out_dir, "features_fp32.bin"))
    edge_index.tofile(os.path.join(out_dir, "norm_edge_index_int64.bin"))
    norm_edge_weights.tofile(os.path.join(out_dir, "norm_edge_weight_fp32.bin"))
    labels.tofile(os.path.join(out_dir, "labels_int64.bin"))
    train_mask.tofile(os.path.join(out_dir, "train_mask_uint8.bin"))
    val_mask.tofile(os.path.join(out_dir, "val_mask_uint8.bin"))
    test_mask.tofile(os.path.join(out_dir, "test_mask_uint8.bin"))
    print("    -> All 7 binary dataset files generated successfully!")

    # 7. Generate Fresh Initial Weights (Seed 2026)
    SEED = 2026
    np.random.seed(SEED)
    H = 16

    std1 = np.sqrt(2.0 / F)
    std2 = np.sqrt(2.0 / H)

    w1 = np.random.normal(0.0, std1, size=(H * F)).astype(np.float32)
    b1 = np.zeros(H, dtype=np.float32)
    w2 = np.random.normal(0.0, std2, size=(C * H)).astype(np.float32)
    b2 = np.zeros(C, dtype=np.float32)

    weight_path = "checkpoints/gcn_initial_weights_seed2026.bin"
    with open(weight_path, "wb") as f:
        f.write(w1.tobytes())
        f.write(b1.tobytes())
        f.write(w2.tobytes())
        f.write(b2.tobytes())

    total_params = len(w1) + len(b1) + len(w2) + len(b2)
    print(f"\n[7] Initial Weights generated with Seed {SEED}:")
    print(f"    Layer 1 (W1: 16x1433, b1: 16): {len(w1) + len(b1)} parameters")
    print(f"    Layer 2 (W2: 7x16,   b2: 7):  {len(w2) + len(b2)} parameters")
    print(f"    Total: {total_params} float32 values ({total_params * 4} bytes)")
    print(f"    Saved to: {weight_path}")
    print("\n==================================================")
    print("                BUILD COMPLETED SUCCESSFULLY!           ")
    print("==================================================")

if __name__ == "__main__":
    build_cora_from_scratch()

