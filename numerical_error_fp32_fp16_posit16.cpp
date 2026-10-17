#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <sw/universal/number/cfloat/cfloat.hpp>
#include <sw/universal/number/posit/posit.hpp>

using namespace std;
using FP32 = sw::universal::fp32;
using FP16 = sw::universal::cfloat<16, 5>;
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

struct Outputs {
    vector<double> support1;
    vector<double> hidden;
    vector<double> support2;
    vector<double> logits;
};

template <typename T>
Outputs forward(
    const vector<float>& xr,
    const vector<float>& edge_weight_raw,
    const vector<int64_t>& edges,
    const vector<float>& saved
) {
    size_t off = 0;
    vector<T> w1(H * F), b1(H), w2(C * H), b2(C);
    for (auto& v : w1) v = T(saved[off++]);
    for (auto& v : b1) v = T(saved[off++]);
    for (auto& v : w2) v = T(saved[off++]);
    for (auto& v : b2) v = T(saved[off++]);

    vector<T> x(N * F), ew(E);
    for (size_t i = 0; i < x.size(); ++i) x[i] = T(xr[i]);
    for (int e = 0; e < E; ++e) ew[e] = T(edge_weight_raw[e]);

    vector<T> support1(N * H, T(0));
    for (int n = 0; n < N; ++n)
        for (int h = 0; h < H; ++h) {
            T sum = T(0);
            for (int f = 0; f < F; ++f)
                sum += x[n * F + f] * w1[h * F + f];
            support1[n * H + h] = sum;
        }

    vector<T> hidden(N * H, T(0));
    for (int e = 0; e < E; ++e) {
        int src = static_cast<int>(edges[e]);
        int dst = static_cast<int>(edges[E + e]);
        for (int h = 0; h < H; ++h)
            hidden[dst * H + h] += ew[e] * support1[src * H + h];
    }
    for (int n = 0; n < N; ++n)
        for (int h = 0; h < H; ++h) {
            T v = hidden[n * H + h] + b1[h];
            hidden[n * H + h] = v > T(0) ? v : T(0);
        }

    vector<T> support2(N * C, T(0));
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c) {
            T sum = T(0);
            for (int h = 0; h < H; ++h)
                sum += hidden[n * H + h] * w2[c * H + h];
            support2[n * C + c] = sum;
        }

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

    Outputs out;
    out.support1.reserve(support1.size());
    out.hidden.reserve(hidden.size());
    out.support2.reserve(support2.size());
    out.logits.reserve(logits.size());
    for (const auto& v : support1) out.support1.push_back(static_cast<double>(v));
    for (const auto& v : hidden) out.hidden.push_back(static_cast<double>(v));
    for (const auto& v : support2) out.support2.push_back(static_cast<double>(v));
    for (const auto& v : logits) out.logits.push_back(static_cast<double>(v));
    return out;
}

void report_error(const string& format, const string& layer,
                  const vector<double>& ref, const vector<double>& got,
                  ofstream& csv) {
    double sum_abs = 0.0, sum_sq = 0.0, max_abs = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double e = fabs(got[i] - ref[i]);
        sum_abs += e;
        sum_sq += e * e;
        max_abs = max(max_abs, e);
    }
    double mae = sum_abs / ref.size();
    double rmse = sqrt(sum_sq / ref.size());
    cout << format << " | " << layer << " | MAE=" << mae
         << " | RMSE=" << rmse << " | MaxAbs=" << max_abs << '\n';
    csv << format << ',' << layer << ',' << ref.size() << ','
        << mae << ',' << rmse << ',' << max_abs << '\n';
}

int main() {
    auto x = load_binary<float>("features_fp32.bin", N * F);
    auto ew = load_binary<float>("norm_edge_weight_fp32.bin", E);
    auto edges = load_binary<int64_t>("norm_edge_index_int64.bin", 2 * E);
    auto saved = load_binary<float>("gcn_trained_weights_fp32.bin",
                                    H * F + H + C * H + C);

    for (int e = 0; e < E; ++e) {
        if (edges[e] < 0 || edges[e] >= N ||
            edges[E + e] < 0 || edges[E + e] >= N) {
            cerr << "Invalid graph edge at " << e << '\n';
            return 1;
        }
    }

    // Reference uses double arithmetic on the same stored FP32 inputs and weights.
    auto ref = forward<double>(x, ew, edges, saved);
    auto f32 = forward<FP32>(x, ew, edges, saved);
    auto f16 = forward<FP16>(x, ew, edges, saved);
    auto p16 = forward<Posit16>(x, ew, edges, saved);

    ofstream csv("numerical_error_fp32_fp16_posit16.csv");
    if (!csv) {
        cerr << "Cannot create numerical_error_fp32_fp16_posit16.csv\n";
        return 1;
    }
    csv << "format,layer,count,mae,rmse,max_absolute_error\n";
    cout << "Numerical error against double-precision reference\n";
    cout << "Shared checkpoint: gcn_trained_weights_fp32.bin\n";
    cout << "All formats use the same stored inputs, graph, and master weights.\n\n";

    for (const string layer : {"support1", "hidden_after_relu", "support2", "final_logits"}) {
        const vector<double>* r = nullptr;
        const vector<double>* a = nullptr;
        const vector<double>* b = nullptr;
        const vector<double>* c = nullptr;
        if (layer == "support1") {
            r=&ref.support1; a=&f32.support1; b=&f16.support1; c=&p16.support1;
        } else if (layer == "hidden_after_relu") {
            r=&ref.hidden; a=&f32.hidden; b=&f16.hidden; c=&p16.hidden;
        } else if (layer == "support2") {
            r=&ref.support2; a=&f32.support2; b=&f16.support2; c=&p16.support2;
        } else {
            r=&ref.logits; a=&f32.logits; b=&f16.logits; c=&p16.logits;
        }
        report_error("FP32", layer, *r, *a, csv);
        report_error("FP16", layer, *r, *b, csv);
        report_error("Posit16", layer, *r, *c, csv);
    }
    cout << "\nSaved CSV: numerical_error_fp32_fp16_posit16.csv\n";
    return 0;
}
