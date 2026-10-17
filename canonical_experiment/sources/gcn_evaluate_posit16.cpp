#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <sw/universal/number/posit/posit.hpp>

using namespace std;
using Posit16 = sw::universal::posit<16, 1>;

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

double val(const Posit16& x) {
    return static_cast<double>(x);
}

void evaluate_split(
    const string& name,
    const vector<uint8_t>& mask,
    const vector<int64_t>& labels,
    const vector<Posit16>& logits
) {
    int count = 0, correct = 0;
    double total_loss = 0.0;

    for (int n = 0; n < N; ++n) {
        if (!mask[n]) continue;

        int y = static_cast<int>(labels[n]);
        if (y < 0 || y >= C) {
            cerr << "Invalid label at node " << n << '\n';
            exit(1);
        }

        int predicted = 0;
        double max_logit = val(logits[n * C]);

        for (int c = 1; c < C; ++c) {
            double z = val(logits[n * C + c]);

            if (z > val(logits[n * C + predicted]))
                predicted = c;

            max_logit = max(max_logit, z);
        }

        if (predicted == y) ++correct;

        double exp_sum = 0.0;
        for (int c = 0; c < C; ++c)
            exp_sum += exp(val(logits[n * C + c]) - max_logit);

        total_loss += max_logit + log(exp_sum)
                    - val(logits[n * C + y]);

        ++count;
    }

    if (count == 0) {
        cerr << "Empty split: " << name << '\n';
        exit(1);
    }

    cout << name
         << " nodes: " << count
         << " | accuracy: " << static_cast<double>(correct) / count
         << " (" << correct << "/" << count << ")"
         << " | cross-entropy: " << total_loss / count << '\n';
}

int main() {
    auto xr = load_binary<float>("features_fp32.bin", N * F);
    auto er = load_binary<float>("norm_edge_weight_fp32.bin", E);
    auto edges = load_binary<int64_t>("norm_edge_index_int64.bin", 2 * E);
    auto labels = load_binary<int64_t>("labels_int64.bin", N);

    auto train_mask = load_binary<uint8_t>("train_mask_uint8.bin", N);
    auto val_mask = load_binary<uint8_t>("val_mask_uint8.bin", N);
    auto test_mask = load_binary<uint8_t>("test_mask_uint8.bin", N);

    // Checkpoint format: W1, b1, W2, b2, each stored as float32.
    auto saved = load_binary<float>(
        "gcn_trained_weights_posit16_100ep.bin", H * F + H + C * H + C
    );

    size_t offset = 0;
    vector<Posit16> w1(H * F), b1(H), w2(C * H), b2(C);

    for (auto& v : w1) v = Posit16(saved[offset++]);
    for (auto& v : b1) v = Posit16(saved[offset++]);
    for (auto& v : w2) v = Posit16(saved[offset++]);
    for (auto& v : b2) v = Posit16(saved[offset++]);

    vector<Posit16> x(N * F), ew(E);

    for (size_t i = 0; i < x.size(); ++i)
        x[i] = Posit16(xr[i]);

    for (int e = 0; e < E; ++e) {
        ew[e] = Posit16(er[e]);

        if (edges[e] < 0 || edges[e] >= N ||
            edges[E + e] < 0 || edges[E + e] >= N) {
            cerr << "Invalid graph edge at " << e << '\n';
            return 1;
        }
    }

    // Layer 1: support1 = X * W1^T
    vector<Posit16> support1(N * H, Posit16(0));

    for (int n = 0; n < N; ++n)
        for (int h = 0; h < H; ++h) {
            Posit16 sum = Posit16(0);
            for (int f = 0; f < F; ++f)
                sum += x[n * F + f] * w1[h * F + f];
            support1[n * H + h] = sum;
        }

    // Aggregate neighbors using the exported normalized graph.
    vector<Posit16> hidden(N * H, Posit16(0));

    for (int e = 0; e < E; ++e) {
        int src = static_cast<int>(edges[e]);
        int dst = static_cast<int>(edges[E + e]);

        for (int h = 0; h < H; ++h)
            hidden[dst * H + h] += ew[e] * support1[src * H + h];
    }

    for (int n = 0; n < N; ++n)
        for (int h = 0; h < H; ++h) {
            Posit16 v = hidden[n * H + h] + b1[h];
            hidden[n * H + h] = v > Posit16(0) ? v : Posit16(0);
        }

    // Layer 2: support2 = hidden * W2^T
    vector<Posit16> support2(N * C, Posit16(0));

    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c) {
            Posit16 sum = Posit16(0);
            for (int h = 0; h < H; ++h)
                sum += hidden[n * H + h] * w2[c * H + h];
            support2[n * C + c] = sum;
        }

    vector<Posit16> logits(N * C, Posit16(0));

    for (int e = 0; e < E; ++e) {
        int src = static_cast<int>(edges[e]);
        int dst = static_cast<int>(edges[E + e]);

        for (int c = 0; c < C; ++c)
            logits[dst * C + c] += ew[e] * support2[src * C + c];
    }

    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c)
            logits[n * C + c] += b2[c];

    cout << "Universal Posit16 GCN evaluation\n";
    cout << "Checkpoint: gcn_trained_weights_posit16_100ep.bin\n";
    cout << "Metrics below use the saved model without training updates.\n";

    evaluate_split("Training", train_mask, labels, logits);
    evaluate_split("Validation", val_mask, labels, logits);
    evaluate_split("Test", test_mask, labels, logits);

    return 0;
}
