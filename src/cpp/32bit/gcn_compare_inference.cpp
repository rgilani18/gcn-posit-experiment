#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <sw/universal/number/cfloat/cfloat.hpp>
#include <sw/universal/number/posit/posit.hpp>

using namespace std;

using FP32 = sw::universal::fp32;
using Posit32 = sw::universal::posit<32, 2>;

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

template <typename T>
double as_double(const T& value) {
    return static_cast<double>(value);
}

template <typename T>
vector<T> forward(
    const vector<float>& raw_x,
    const vector<float>& raw_ew,
    const vector<int64_t>& edges,
    const vector<float>& raw_weights
) {
    vector<T> x(N * F), ew(E);
    vector<T> w1(H * F), b1(H), w2(C * H), b2(C);

    for (size_t i = 0; i < x.size(); ++i)
        x[i] = T(raw_x[i]);

    for (int e = 0; e < E; ++e)
        ew[e] = T(raw_ew[e]);

    size_t offset = 0;
    for (auto& v : w1) v = T(raw_weights[offset++]);
    for (auto& v : b1) v = T(raw_weights[offset++]);
    for (auto& v : w2) v = T(raw_weights[offset++]);
    for (auto& v : b2) v = T(raw_weights[offset++]);

    // First linear transform: X * W1^T
    vector<T> support1(N * H, T(0));

    for (int n = 0; n < N; ++n)
        for (int h = 0; h < H; ++h) {
            T sum = T(0);
            for (int f = 0; f < F; ++f)
                sum += x[n * F + f] * w1[h * F + f];
            support1[n * H + h] = sum;
        }

    // First normalized graph aggregation.
    vector<T> hidden(N * H, T(0));

    for (int e = 0; e < E; ++e) {
        int src = static_cast<int>(edges[e]);
        int dst = static_cast<int>(edges[E + e]);

        if (src < 0 || src >= N || dst < 0 || dst >= N) {
            cerr << "Invalid graph edge at " << e << '\n';
            exit(1);
        }

        for (int h = 0; h < H; ++h)
            hidden[dst * H + h] += ew[e] * support1[src * H + h];
    }

    // Bias and ReLU.
    for (int n = 0; n < N; ++n)
        for (int h = 0; h < H; ++h) {
            T v = hidden[n * H + h] + b1[h];
            hidden[n * H + h] = v > T(0) ? v : T(0);
        }

    // Second linear transform: hidden * W2^T
    vector<T> support2(N * C, T(0));

    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c) {
            T sum = T(0);
            for (int h = 0; h < H; ++h)
                sum += hidden[n * H + h] * w2[c * H + h];
            support2[n * C + c] = sum;
        }

    // Second normalized graph aggregation.
    vector<T> logits(N * C, T(0));

    for (int e = 0; e < E; ++e) {
        int src = static_cast<int>(edges[e]);
        int dst = static_cast<int>(edges[E + e]);

        for (int c = 0; c < C; ++c)
            logits[dst * C + c] += ew[e] * support2[src * C + c];
    }

    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c)
            logits[n * C + c] += b2[c];

    return logits;
}

template <typename T>
void evaluate_split(
    const string& name,
    const vector<uint8_t>& mask,
    const vector<int64_t>& labels,
    const vector<T>& logits
) {
    int count = 0, correct = 0;
    double total_loss = 0.0;

    for (int n = 0; n < N; ++n) {
        if (!mask[n]) continue;

        int y = static_cast<int>(labels[n]);
        int predicted = 0;
        double max_logit = as_double(logits[n * C]);

        for (int c = 1; c < C; ++c) {
            double z = as_double(logits[n * C + c]);
            if (z > as_double(logits[n * C + predicted]))
                predicted = c;
            max_logit = max(max_logit, z);
        }

        if (predicted == y) ++correct;

        double exp_sum = 0.0;
        for (int c = 0; c < C; ++c)
            exp_sum += exp(as_double(logits[n * C + c]) - max_logit);

        total_loss += max_logit + log(exp_sum)
                    - as_double(logits[n * C + y]);
        ++count;
    }

    cout << name << ": accuracy="
         << static_cast<double>(correct) / count
         << " (" << correct << "/" << count << ")"
         << ", cross-entropy=" << total_loss / count << '\n';
}

void compare_logits(
    const string& name,
    const vector<double>& reference,
    const vector<double>& actual,
    const vector<uint8_t>* mask = nullptr
) {
    double max_error = 0.0;
    double sum_sq = 0.0;
    size_t count = 0;

    for (int n = 0; n < N; ++n) {
        if (mask && !(*mask)[n]) continue;

        for (int c = 0; c < C; ++c) {
            size_t i = n * C + c;
            double error = abs(actual[i] - reference[i]);
            max_error = max(max_error, error);
            sum_sq += error * error;
            ++count;
        }
    }

    cout << name
         << " | max absolute error=" << max_error
         << " | RMSE=" << sqrt(sum_sq / count)
         << " | compared logits=" << count << '\n';
}

template <typename T>
vector<double> convert_logits(const vector<T>& logits) {
    vector<double> result(logits.size());
    for (size_t i = 0; i < logits.size(); ++i)
        result[i] = as_double(logits[i]);
    return result;
}

int main(int argc, char* argv[]) {
    const string weights_filename = (argc > 1) ? argv[1] : "gcn_trained_weights_fp32_500ep.bin";
    cout << setprecision(12);

    auto x = load_binary<float>("features_fp32.bin", N * F);
    auto ew = load_binary<float>("norm_edge_weight_fp32.bin", E);
    auto edges = load_binary<int64_t>("norm_edge_index_int64.bin", 2 * E);
    auto labels = load_binary<int64_t>("labels_int64.bin", N);
    auto train_mask = load_binary<uint8_t>("train_mask_uint8.bin", N);
    auto val_mask = load_binary<uint8_t>("val_mask_uint8.bin", N);
    auto test_mask = load_binary<uint8_t>("test_mask_uint8.bin", N);

    auto weights = load_binary<float>(
        weights_filename, H * F + H + C * H + C
    );

    cout << "GCN inference comparison: FP32 vs Posit32 vs double\n";
    cout << "Weights: " << weights_filename << "\n";
    cout << "No training or weight updates are performed.\n\n";

    auto fp_logits = forward<FP32>(x, ew, edges, weights);
    auto posit_logits = forward<Posit32>(x, ew, edges, weights);
    auto double_logits = forward<double>(x, ew, edges, weights);

    auto fp = convert_logits(fp_logits);
    auto posit = convert_logits(posit_logits);

    cout << "\n--- FP32 metrics ---\n";
    evaluate_split("Train", train_mask, labels, fp_logits);
    evaluate_split("Validation", val_mask, labels, fp_logits);
    evaluate_split("Test", test_mask, labels, fp_logits);

    cout << "\n--- Posit32 metrics ---\n";
    evaluate_split("Train", train_mask, labels, posit_logits);
    evaluate_split("Validation", val_mask, labels, posit_logits);
    evaluate_split("Test", test_mask, labels, posit_logits);

    cout << "\n--- Logit errors against double arithmetic ---\n";
    compare_logits("FP32 all nodes", double_logits, fp);
    compare_logits("Posit32 all nodes", double_logits, posit);
    compare_logits("FP32 test nodes", double_logits, fp, &test_mask);
    compare_logits("Posit32 test nodes", double_logits, posit, &test_mask);

    return 0;
}
