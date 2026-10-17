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

constexpr int N = 2708;
constexpr int F = 1433;
constexpr int H = 16;
constexpr int C = 7;
constexpr int E = 13264;

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

    if (file.gcount() !=
        static_cast<streamsize>(count * sizeof(T))) {
        cerr << "Incorrect file size: " << filename << '\n';
        exit(1);
    }

    return data;
}

int main() {
    cout << "Fresh Universal FP32 GCN training setup\n";

    auto x_raw = load_binary<float>("features_fp32.bin", N * F);
    auto edge = load_binary<int64_t>("norm_edge_index_int64.bin", 2 * E);
    auto ew_raw = load_binary<float>("norm_edge_weight_fp32.bin", E);
    auto labels = load_binary<int64_t>("labels_int64.bin", N);

    auto train_mask =
        load_binary<uint8_t>("train_mask_uint8.bin", N);
    auto val_mask =
        load_binary<uint8_t>("val_mask_uint8.bin", N);
    auto test_mask =
        load_binary<uint8_t>("test_mask_uint8.bin", N);

    vector<FP32> x(N * F), ew(E);
    for (size_t i = 0; i < x.size(); ++i)
        x[i] = FP32(x_raw[i]);

    for (int e = 0; e < E; ++e)
        ew[e] = FP32(ew_raw[e]);

    // Fresh weights; do not load the previous PyTorch checkpoint.
    mt19937 rng(42);
    normal_distribution<float> dist(0.0f, 1.0f);

    vector<FP32> w1(H * F), b1(H, FP32(0));
    vector<FP32> w2(C * H), b2(C, FP32(0));

    const float limit1 = sqrt(2.0f / F);
    const float limit2 = sqrt(2.0f / H);

    for (auto& w : w1) w = FP32(dist(rng) * limit1);
    for (auto& w : w2) w = FP32(dist(rng) * limit2);

    int train_count = 0, val_count = 0, test_count = 0;
    for (int n = 0; n < N; ++n) {
        train_count += train_mask[n] != 0;
        val_count += val_mask[n] != 0;
        test_count += test_mask[n] != 0;
    }

    cout << "Features loaded: " << x.size() << '\n';
    cout << "Edges loaded: " << edge.size() / 2 << '\n';
    cout << "Training nodes: " << train_count << '\n';
    cout << "Validation nodes: " << val_count << '\n';
    cout << "Test nodes: " << test_count << '\n';
    cout << "Fresh weights initialized: "
         << w1.size() + w2.size() + b1.size() + b2.size()
         << '\n';
    cout << "Setup only; no training performed yet.\n";

    return 0;
}
