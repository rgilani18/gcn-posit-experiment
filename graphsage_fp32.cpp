
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

constexpr int N = 7650;
constexpr int F = 745;
constexpr int H = 64;
constexpr int C = 8;
constexpr int E = 238162;
constexpr int T = 1530;

std::vector<float> read_f32(const char* path, size_t count) {
    std::vector<float> v(count);
    std::ifstream f(path, std::ios::binary);
    if (!f.read(reinterpret_cast<char*>(v.data()),
                count * sizeof(float))) {
        std::cerr << "Failed reading " << path << "\n";
        std::exit(1);
    }
    return v;
}

std::vector<int64_t> read_i64(const char* path, size_t count) {
    std::vector<int64_t> v(count);
    std::ifstream f(path, std::ios::binary);
    if (!f.read(reinterpret_cast<char*>(v.data()),
                count * sizeof(int64_t))) {
        std::cerr << "Failed reading " << path << "\n";
        std::exit(1);
    }
    return v;
}

int main() {
    auto x = read_f32("amazon_photo_features_fp32.bin", N * F);
    auto src = read_i64("amazon_photo_edge_src_int64.bin", E);
    auto dst = read_i64("amazon_photo_edge_dst_int64.bin", E);

    auto w1l = read_f32("conv1_lin_l_weight_fp32.bin", H * F);
    auto b1l = read_f32("conv1_lin_l_bias_fp32.bin", H);
    auto w1r = read_f32("conv1_lin_r_weight_fp32.bin", H * F);
    auto w2l = read_f32("conv2_lin_l_weight_fp32.bin", C * H);
    auto b2l = read_f32("conv2_lin_l_bias_fp32.bin", C);
    auto w2r = read_f32("conv2_lin_r_weight_fp32.bin", C * H);

    auto test_idx = read_i64("amazon_photo_test_indices_int64.bin", T);
    auto labels = read_i64("amazon_photo_test_labels_int64.bin", T);

    // Mean aggregation: aggregate source-node features at destinations.
    std::vector<float> agg1(N * F, 0.0f);
    std::vector<int> degree(N, 0);

    for (int e = 0; e < E; ++e) {
        int s = static_cast<int>(src[e]);
        int d = static_cast<int>(dst[e]);
        ++degree[d];
        for (int j = 0; j < F; ++j)
            agg1[d * F + j] += x[s * F + j];
    }

    for (int i = 0; i < N; ++i)
        if (degree[i] > 0)
            for (int j = 0; j < F; ++j)
                agg1[i * F + j] /= degree[i];

    // Layer 1: lin_l(mean(neighbors)) + lin_r(self), then ReLU.
    std::vector<float> h1(N * H, 0.0f);

    for (int i = 0; i < N; ++i) {
        for (int o = 0; o < H; ++o) {
            float neighbor_sum = b1l[o];
            float root_sum = 0.0f;

            for (int j = 0; j < F; ++j) {
                neighbor_sum += w1l[o * F + j] * agg1[i * F + j];
                root_sum += w1r[o * F + j] * x[i * F + j];
            }

            float v = neighbor_sum + root_sum;
            h1[i * H + o] = std::max(0.0f, v);
        }
    }

    // Layer 2: mean aggregation over layer-1 features.
    std::vector<float> agg2(N * H, 0.0f);
    std::fill(degree.begin(), degree.end(), 0);

    for (int e = 0; e < E; ++e) {
        int s = static_cast<int>(src[e]);
        int d = static_cast<int>(dst[e]);
        ++degree[d];
        for (int j = 0; j < H; ++j)
            agg2[d * H + j] += h1[s * H + j];
    }

    for (int i = 0; i < N; ++i)
        if (degree[i] > 0)
            for (int j = 0; j < H; ++j)
                agg2[i * H + j] /= degree[i];

    // Layer 2 outputs logits; no ReLU or dropout in evaluation.
    std::vector<float> logits(N * C, 0.0f);

    for (int i = 0; i < N; ++i) {
        for (int o = 0; o < C; ++o) {
            float neighbor_sum = b2l[o];
            float root_sum = 0.0f;

            for (int j = 0; j < H; ++j) {
                neighbor_sum += w2l[o * H + j] * agg2[i * H + j];
                root_sum += w2r[o * H + j] * h1[i * H + j];
            }

            logits[i * C + o] = neighbor_sum + root_sum;
        }
    }

    std::ofstream out("amazon_photo_test_logits_cpp_fp32.bin",
                      std::ios::binary);
    int correct = 0;

    for (int k = 0; k < T; ++k) {
        int i = static_cast<int>(test_idx[k]);
        int best = 0;

        for (int c = 1; c < C; ++c)
            if (logits[i * C + c] > logits[i * C + best])
                best = c;

        if (best == labels[k]) ++correct;

        out.write(reinterpret_cast<const char*>(&logits[i * C]),
                  C * sizeof(float));
    }

    std::cout << "C++ FP32 test accuracy: "
              << static_cast<double>(correct) / T << "\n";
    std::cout << "Saved C++ test logits.\n";
}
