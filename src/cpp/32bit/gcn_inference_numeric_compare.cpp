#include <algorithm>
#include <chrono>
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
#include <sw/universal/number/dd/dd.hpp>

using namespace std;

using FP32   = sw::universal::fp32;
using Posit32 = sw::universal::posit<32, 2>;
using DD     = sw::universal::dd;

constexpr int N = 2708;
constexpr int F = 1433;
constexpr int H = 16;
constexpr int C = 7;
constexpr int E = 13264;

template<typename T>
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

struct Inputs {
    vector<float> features, edge_weights, saved_weights;
    vector<int64_t> edges, labels;
    vector<uint8_t> train_mask, val_mask, test_mask;
};

Inputs load_inputs() {
    Inputs d;
    d.features = load_binary<float>("features_fp32.bin", N * F);
    d.edge_weights =
        load_binary<float>("norm_edge_weight_fp32.bin", E);
    d.edges =
        load_binary<int64_t>("norm_edge_index_int64.bin", 2 * E);
    d.labels = load_binary<int64_t>("labels_int64.bin", N);

    d.train_mask = load_binary<uint8_t>("train_mask_uint8.bin", N);
    d.val_mask = load_binary<uint8_t>("val_mask_uint8.bin", N);
    d.test_mask = load_binary<uint8_t>("test_mask_uint8.bin", N);

    d.saved_weights = load_binary<float>(
        "gcn_trained_weights_fp32.bin", H * F + H + C * H + C
    );

    for (int e = 0; e < E; ++e) {
        if (d.edges[e] < 0 || d.edges[e] >= N ||
            d.edges[E + e] < 0 || d.edges[E + e] >= N) {
            cerr << "Invalid graph edge: " << e << '\n';
            exit(1);
        }
    }

    return d;
}

struct Result {
    vector<double> hidden;
    vector<double> logits;
    double seconds = 0.0;
};

template<typename T>
Result run_inference(const Inputs& d) {
    size_t off = 0;
    vector<T> w1(H * F), b1(H), w2(C * H), b2(C);

    for (auto& v : w1) v = T(d.saved_weights[off++]);
    for (auto& v : b1) v = T(d.saved_weights[off++]);
    for (auto& v : w2) v = T(d.saved_weights[off++]);
    for (auto& v : b2) v = T(d.saved_weights[off++]);

    vector<T> x(N * F), ew(E);
    for (size_t i = 0; i < x.size(); ++i)
        x[i] = T(d.features[i]);

    for (int e = 0; e < E; ++e)
        ew[e] = T(d.edge_weights[e]);

    const auto start = chrono::steady_clock::now();

    // Layer 1: support1 = X * W1^T
    vector<T> support1(N * H, T(0));

    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < H; ++h) {
            T sum = T(0);
            for (int f = 0; f < F; ++f)
                sum += x[n * F + f] * w1[h * F + f];
            support1[n * H + h] = sum;
        }
    }

    // First graph aggregation
    vector<T> hidden(N * H, T(0));

    for (int e = 0; e < E; ++e) {
        int src = static_cast<int>(d.edges[e]);
        int dst = static_cast<int>(d.edges[E + e]);

        for (int h = 0; h < H; ++h)
            hidden[dst * H + h] +=
                ew[e] * support1[src * H + h];
    }

    // Bias + ReLU
    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < H; ++h) {
            T v = hidden[n * H + h] + b1[h];
            hidden[n * H + h] = v > T(0) ? v : T(0);
        }
    }

    // Layer 2: support2 = hidden * W2^T
    vector<T> support2(N * C, T(0));

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            T sum = T(0);
            for (int h = 0; h < H; ++h)
                sum += hidden[n * H + h] * w2[c * H + h];
            support2[n * C + c] = sum;
        }
    }

    // Second graph aggregation
    vector<T> logits(N * C, T(0));

    for (int e = 0; e < E; ++e) {
        int src = static_cast<int>(d.edges[e]);
        int dst = static_cast<int>(d.edges[E + e]);

        for (int c = 0; c < C; ++c)
            logits[dst * C + c] +=
                ew[e] * support2[src * C + c];
    }

    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c)
            logits[n * C + c] += b2[c];

    const auto stop = chrono::steady_clock::now();

    Result r;
    r.seconds = chrono::duration<double>(stop - start).count();

    // Convert only for reporting and comparison.
    r.hidden.resize(hidden.size());
    r.logits.resize(logits.size());

    for (size_t i = 0; i < hidden.size(); ++i)
        r.hidden[i] = static_cast<double>(hidden[i]);

    for (size_t i = 0; i < logits.size(); ++i)
        r.logits[i] = static_cast<double>(logits[i]);

    return r;
}

void report_error(const string& name,
                  const vector<double>& values,
                  const vector<double>& reference) {
    double max_error = 0.0;
    double total_error = 0.0;

    for (size_t i = 0; i < values.size(); ++i) {
        double error = abs(values[i] - reference[i]);
        max_error = max(max_error, error);
        total_error += error;
    }

    cout << name
         << " | Max absolute error: " << max_error
         << " | MAE: " << total_error / values.size()
         << '\n';
}

void report_split(const string& name,
                  const vector<uint8_t>& mask,
                  const vector<int64_t>& labels,
                  const vector<double>& logits) {
    int count = 0, correct = 0;
    double total_loss = 0.0;

    for (int n = 0; n < N; ++n) {
        if (!mask[n]) continue;

        int predicted = 0;
        double max_logit = logits[n * C];

        for (int c = 1; c < C; ++c) {
            double z = logits[n * C + c];
            if (z > logits[n * C + predicted])
                predicted = c;
            max_logit = max(max_logit, z);
        }

        if (predicted == labels[n]) ++correct;

        double exp_sum = 0.0;
        for (int c = 0; c < C; ++c)
            exp_sum += exp(logits[n * C + c] - max_logit);

        total_loss += max_logit + log(exp_sum)
                    - logits[n * C + labels[n]];
        ++count;
    }

    cout << name << " accuracy: "
         << static_cast<double>(correct) / count
         << " (" << correct << "/" << count << ")"
         << " | cross-entropy: " << total_loss / count
         << '\n';
}

void report_metrics(const string& name,
                    const Inputs& d,
                    const Result& r) {
    cout << "\n[" << name << "]\n";
    cout << "Forward-pass runtime (seconds): "
         << r.seconds << '\n';

    report_split("Train", d.train_mask, d.labels, r.logits);
    report_split("Validation", d.val_mask, d.labels, r.logits);
    report_split("Test", d.test_mask, d.labels, r.logits);
}

int main() {
    cout << setprecision(12);

    Inputs d = load_inputs();

    cout << "Checkpoint: FP32-trained 500-epoch model\n";
    cout << "Arithmetic: Universal FP32, Posit32, and dd\n";
    cout << "Parameters loaded in order: W1, b1, W2, b2\n";

    Result ref = run_inference<DD>(d);
    Result fp  = run_inference<FP32>(d);
    Result po  = run_inference<Posit32>(d);

    cout << "\n=== Layer-wise numerical error vs dd reference ===\n";
    report_error("FP32 hidden layer", fp.hidden, ref.hidden);
    report_error("Posit32 hidden layer", po.hidden, ref.hidden);
    report_error("FP32 logits", fp.logits, ref.logits);
    report_error("Posit32 logits", po.logits, ref.logits);

    cout << "\n=== Model metrics ===\n";
    report_metrics("dd reference", d, ref);
    report_metrics("Universal FP32", d, fp);
    report_metrics("Universal Posit32", d, po);

    cout << "\nNote: dd is a higher-precision reference, not an "
            "exact mathematical oracle.\n";
    cout << "Runtime is a single forward-pass measurement; "
            "repeat runs are needed for reliable timing comparisons.\n";
    return 0;
}
