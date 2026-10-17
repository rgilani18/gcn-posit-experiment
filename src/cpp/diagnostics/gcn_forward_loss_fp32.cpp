#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <random>
#include <string>

#include <universal/number/cfloat/cfloat.hpp>

using namespace std;
using FP32 = sw::universal::fp32;

constexpr int N = 2708, F = 1433, H = 16, C = 7, E = 13264;

template <typename T>
vector<T> load_binary(const string& filename, size_t count) {
    vector<T> data(count);
    ifstream file(filename, ios::binary);
    if (!file) {
        cerr << "Cannot open " << filename << '\n';
        exit(1);
    }
    file.read(reinterpret_cast<char*>(data.data()),
              static_cast<streamsize>(count * sizeof(T)));
    if (file.gcount() != static_cast<streamsize>(count * sizeof(T))) {
        cerr << "Incorrect file size: " << filename << '\n';
        exit(1);
    }
    return data;
}

int main() {
    auto xr = load_binary<float>("features_fp32.bin", N * F);
    auto er = load_binary<float>("norm_edge_weight_fp32.bin", E);
    auto edges = load_binary<int64_t>("norm_edge_index_int64.bin", 2 * E);
    auto labels = load_binary<int64_t>("labels_int64.bin", N);
    auto mask = load_binary<uint8_t>("train_mask_uint8.bin", N);

    vector<FP32> x(N * F), ew(E), w1(H * F), b1(H, FP32(0));
    vector<FP32> w2(C * H), b2(C, FP32(0));

    for (size_t i = 0; i < x.size(); ++i) x[i] = FP32(xr[i]);
    for (int e = 0; e < E; ++e) ew[e] = FP32(er[e]);

    mt19937 rng(42);
    normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& w : w1) w = FP32(dist(rng) * sqrt(2.0f / F));
    for (auto& w : w2) w = FP32(dist(rng) * sqrt(2.0f / H));

    // Layer 1: support = X * W1^T
    vector<FP32> support1(N * H, FP32(0));
    for (int n = 0; n < N; ++n)
        for (int h = 0; h < H; ++h) {
            FP32 sum = FP32(0);
            for (int f = 0; f < F; ++f)
                sum += x[n * F + f] * w1[h * F + f];
            support1[n * H + h] = sum;
        }

    // Aggregate neighbors, then bias + ReLU.
    vector<FP32> hidden(N * H, FP32(0));
    for (int e = 0; e < E; ++e) {
        int src = static_cast<int>(edges[e]);
        int dst = static_cast<int>(edges[E + e]);
        if (src < 0 || src >= N || dst < 0 || dst >= N) {
            cerr << "Invalid edge index\n";
            return 1;
        }
        for (int h = 0; h < H; ++h)
            hidden[dst * H + h] += ew[e] * support1[src * H + h];
    }
    for (int n = 0; n < N; ++n)
        for (int h = 0; h < H; ++h) {
            FP32 v = hidden[n * H + h] + b1[h];
            hidden[n * H + h] = v > FP32(0) ? v : FP32(0);
        }

    // Layer 2: support = hidden * W2^T
    vector<FP32> support2(N * C, FP32(0));
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c) {
            FP32 sum = FP32(0);
            for (int h = 0; h < H; ++h)
                sum += hidden[n * H + h] * w2[c * H + h];
            support2[n * C + c] = sum;
        }

    vector<FP32> logits(N * C, FP32(0));
    for (int e = 0; e < E; ++e) {
        int src = static_cast<int>(edges[e]);
        int dst = static_cast<int>(edges[E + e]);
        for (int c = 0; c < C; ++c)
            logits[dst * C + c] += ew[e] * support2[src * C + c];
    }
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c)
            logits[n * C + c] += b2[c];

    // Mean cross-entropy over training nodes; stable log-sum-exp.
    double loss = 0.0;
    int count = 0;
    for (int n = 0; n < N; ++n) {
        if (!mask[n]) continue;

        int y = static_cast<int>(labels[n]);
        if (y < 0 || y >= C) {
            cerr << "Invalid label\n";
            return 1;
        }

        double maxlogit = static_cast<double>(logits[n * C]);
        for (int c = 1; c < C; ++c)
            maxlogit = max(maxlogit, static_cast<double>(logits[n * C + c]));

        double sumexp = 0.0;
        for (int c = 0; c < C; ++c)
            sumexp += exp(static_cast<double>(logits[n * C + c]) - maxlogit);

        loss += maxlogit + log(sumexp)
              - static_cast<double>(logits[n * C + y]);
        ++count;
    }

    cout << "Universal FP32 forward/loss check\n";
    cout << "Training nodes: " << count << '\n';
    cout << "Initial mean cross-entropy: " << loss / count << '\n';
    cout << "No backpropagation or weight updates yet.\n";
    return 0;
}
