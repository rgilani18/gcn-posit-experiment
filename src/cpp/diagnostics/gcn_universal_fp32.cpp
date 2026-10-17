#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cstdlib>

#include <universal/number/cfloat/cfloat.hpp>

using namespace std;
using FP32 = sw::universal::fp32;

constexpr int N = 2708;
constexpr int F = 1433;
constexpr int H = 16;
constexpr int C = 7;
constexpr int E = 13264;

template <typename T>
vector<T> load_binary(const string& filename, size_t count)
{
    vector<T> data(count);
    ifstream file(filename, ios::binary);

    if (!file) {
        cerr << "Cannot open " << filename << '\n';
        exit(1);
    }

    file.read(reinterpret_cast<char*>(data.data()),
              count * sizeof(T));

    if (file.gcount() != static_cast<streamsize>(count * sizeof(T))) {
        cerr << "Incorrect file size: " << filename << '\n';
        exit(1);
    }

    return data;
}

int main()
{
    cout << "Universal Library FP32 GCN\n";
    cout << "Loading data...\n";

    // Input data is stored as ordinary binary32.
    auto x_raw = load_binary<float>("features_fp32.bin", N * F);
    auto w1_raw = load_binary<float>("conv1_weight_fp32.bin", H * F);
    auto b1_raw = load_binary<float>("conv1_bias_fp32.bin", H);
    auto w2_raw = load_binary<float>("conv2_weight_fp32.bin", C * H);
    auto b2_raw = load_binary<float>("conv2_bias_fp32.bin", C);
    auto ew_raw = load_binary<float>("norm_edge_weight_fp32.bin", E);
    auto edge = load_binary<long long>("norm_edge_index_int64.bin", 2 * E);
    auto labels = load_binary<long long>("labels_int64.bin", N);
    auto mask = load_binary<unsigned char>("test_mask_uint8.bin", N);
    auto golden = load_binary<float>("golden_logits_fp32.bin", N * C);

    // Convert numeric operands to the Universal FP32 type.
    vector<FP32> x(N * F), w1(H * F), b1(H);
    vector<FP32> w2(C * H), b2(C), ew(E);

    for (size_t i = 0; i < x.size(); ++i) x[i] = FP32(x_raw[i]);
    for (size_t i = 0; i < w1.size(); ++i) w1[i] = FP32(w1_raw[i]);
    for (size_t i = 0; i < b1.size(); ++i) b1[i] = FP32(b1_raw[i]);
    for (size_t i = 0; i < w2.size(); ++i) w2[i] = FP32(w2_raw[i]);
    for (size_t i = 0; i < b2.size(); ++i) b2[i] = FP32(b2_raw[i]);
    for (size_t i = 0; i < ew.size(); ++i) ew[i] = FP32(ew_raw[i]);

    cout << "Data loaded and converted to Universal FP32.\n";

    // Layer 1: X * W1^T
    vector<FP32> support1(N * H, FP32(0));

    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < H; ++h) {
            FP32 sum = FP32(0);
            for (int f = 0; f < F; ++f)
                sum += x[n * F + f] * w1[h * F + f];
            support1[n * H + h] = sum;
        }
    }

    // Aggregate using the precomputed normalized edge weights.
    vector<FP32> hidden(N * H, FP32(0));

    for (int e = 0; e < E; ++e) {
        int src = static_cast<int>(edge[e]);
        int dst = static_cast<int>(edge[E + e]);

        if (src < 0 || src >= N || dst < 0 || dst >= N) {
            cerr << "Invalid node index in edge data.\n";
            return 1;
        }

        for (int h = 0; h < H; ++h)
            hidden[dst * H + h] += ew[e] * support1[src * H + h];
    }

    // Bias and ReLU
    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < H; ++h) {
            FP32 value = hidden[n * H + h] + b1[h];
            hidden[n * H + h] = (value > FP32(0)) ? value : FP32(0);
        }
    }

    // Layer 2: H * W2^T
    vector<FP32> support2(N * C, FP32(0));

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            FP32 sum = FP32(0);
            for (int h = 0; h < H; ++h)
                sum += hidden[n * H + h] * w2[c * H + h];
            support2[n * C + c] = sum;
        }
    }

    // Second graph aggregation
    vector<FP32> logits(N * C, FP32(0));

    for (int e = 0; e < E; ++e) {
        int src = static_cast<int>(edge[e]);
        int dst = static_cast<int>(edge[E + e]);

        for (int c = 0; c < C; ++c)
            logits[dst * C + c] += ew[e] * support2[src * C + c];
    }

    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c)
            logits[n * C + c] += b2[c];

    // Test accuracy and comparison against the saved reference.
    int correct = 0, total = 0;
    double max_error = 0.0, sum_error = 0.0;

    for (int n = 0; n < N; ++n) {
        if (!mask[n]) continue;

        int best = 0;
        for (int c = 1; c < C; ++c)
            if (logits[n * C + c] > logits[n * C + best])
                best = c;

        if (best == labels[n]) ++correct;
        ++total;
    }

    for (size_t i = 0; i < logits.size(); ++i) {
        double err = std::fabs(static_cast<double>(logits[i]) - golden[i]);
        if (err > max_error) max_error = err;
        sum_error += err;
    }

    cout << fixed << setprecision(9);
    cout << "Universal FP32 test accuracy: "
         << static_cast<double>(correct) / total << '\n';
    cout << "Correct: " << correct << " / " << total << '\n';
    cout << "Maximum absolute logit error: " << max_error << '\n';
    cout << "Mean absolute logit error: "
         << sum_error / logits.size() << '\n';

    // Save final Universal FP32 logits for comparison with the long-double reference.
    {
        ofstream output("logits_fp32.bin", ios::binary);
        if (!output) {
            cerr << "Could not create logits_fp32.bin\n";
            return 1;
        }

        for (const auto& value : logits) {
            float stored_value = static_cast<float>(value);
            output.write(reinterpret_cast<const char*>(&stored_value),
                         sizeof(stored_value));
        }

        if (!output) {
            cerr << "Failed while writing logits_fp32.bin\n";
            return 1;
        }
    }
    cout << "Saved: logits_fp32.bin" << '\n';

    return 0;
}
