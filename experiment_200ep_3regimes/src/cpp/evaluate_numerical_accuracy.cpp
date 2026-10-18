#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// Pure Stillwater Universal (NO PyTorch)
#include <universal/number/cfloat/cfloat.hpp>
#include <universal/number/posit/posit.hpp>

using namespace std;

using LongDouble  = long double;
using FP32_Uni    = sw::universal::cfloat<32, 8>;
using FP16_Uni    = sw::universal::cfloat<16, 5>;
using Posit16_Uni = sw::universal::posit<16, 1>;

constexpr int N = 2708;
constexpr int F = 1433;
constexpr int H = 16;
constexpr int C = 7;
constexpr int E = 13264;

template <typename T>
vector<T> load_binary(const string& filename, size_t count) {
    vector<T> data(count);
    ifstream file(filename, ios::binary);
    if (!file) { cerr << "Error opening " << filename << '\n'; exit(1); }
    file.read(reinterpret_cast<char*>(data.data()), count * sizeof(T));
    return data;
}

// 2-Layer GCN Forward Pass (identical code path for all types)
template <typename T>
vector<double> run_2layer_forward(
    const vector<float>& xr, const vector<float>& er,
    const vector<int64_t>& edges,
    const vector<float>& raw_w
) {
    vector<T> x(N * F), ew(E);
    for (size_t i = 0; i < x.size(); ++i) x[i] = T(xr[i]);
    for (int e = 0; e < E; ++e) ew[e] = T(er[e]);

    vector<T> w1(H * F), b1(H), w2(C * H), b2(C);
    size_t off = 0;
    for (size_t i = 0; i < w1.size(); ++i) w1[i] = T(raw_w[off++]);
    for (size_t i = 0; i < b1.size(); ++i) b1[i] = T(raw_w[off++]);
    for (size_t i = 0; i < w2.size(); ++i) w2[i] = T(raw_w[off++]);
    for (size_t i = 0; i < b2.size(); ++i) b2[i] = T(raw_w[off++]);

    // Layer 1: X * W1^T
    vector<T> s1(N * H, T(0));
    for (int n = 0; n < N; ++n)
        for (int h = 0; h < H; ++h)
            for (int f = 0; f < F; ++f)
                s1[n * H + h] += x[n * F + f] * w1[h * F + f];

    // Graph Aggregation 1
    vector<T> agg1(N * H, T(0));
    for (int e = 0; e < E; ++e) {
        int src = static_cast<int>(edges[e]), dst = static_cast<int>(edges[E + e]);
        for (int h = 0; h < H; ++h) agg1[dst * H + h] += ew[e] * s1[src * H + h];
    }

    // Bias + ReLU
    vector<T> h_act(N * H);
    for (int n = 0; n < N; ++n)
        for (int h = 0; h < H; ++h) {
            T v = agg1[n * H + h] + b1[h];
            h_act[n * H + h] = v > T(0) ? v : T(0);
        }

    // Layer 2: H * W2^T
    vector<T> s2(N * C, T(0));
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c)
            for (int h = 0; h < H; ++h)
                s2[n * C + c] += h_act[n * H + h] * w2[c * H + h];

    // Graph Aggregation 2 + Bias
    vector<T> logits(N * C, T(0));
    for (int e = 0; e < E; ++e) {
        int src = static_cast<int>(edges[e]), dst = static_cast<int>(edges[E + e]);
        for (int c = 0; c < C; ++c) logits[dst * C + c] += ew[e] * s2[src * C + c];
    }
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c) logits[n * C + c] += b2[c];

    vector<double> out(N * C);
    for (size_t i = 0; i < logits.size(); ++i) out[i] = static_cast<double>(logits[i]);
    return out;
}

struct ErrorMetrics {
    double mae, rmse, max_err;
    double median_err;
};

ErrorMetrics compute_errors(const vector<double>& candidate, const vector<double>& reference) {
    size_t count = candidate.size();
    double sum_abs = 0.0, sum_sq = 0.0, max_err = 0.0;
    vector<double> errs(count);

    for (size_t i = 0; i < count; ++i) {
        double diff = fabs(candidate[i] - reference[i]);
        errs[i] = diff;
        sum_abs += diff;
        sum_sq += diff * diff;
        max_err = max(max_err, diff);
    }
    sort(errs.begin(), errs.end());
    double median = errs[count / 2];
    return {sum_abs / count, sqrt(sum_sq / count), max_err, median};
}

// Per-node error distribution: for each node, compute the L2 error across C logit dimensions
void save_per_node_errors(
    const string& filename,
    const vector<double>& fp16_logits,
    const vector<double>& pos16_logits,
    const vector<double>& ref_logits
) {
    ofstream csv(filename);
    csv << "node,fp16_l2_error,posit16_l2_error,fp16_max_error,posit16_max_error\n";
    for (int n = 0; n < N; ++n) {
        double fp16_l2 = 0.0, pos16_l2 = 0.0;
        double fp16_max = 0.0, pos16_max = 0.0;
        for (int c = 0; c < C; ++c) {
            double d1 = fabs(fp16_logits[n * C + c] - ref_logits[n * C + c]);
            double d2 = fabs(pos16_logits[n * C + c] - ref_logits[n * C + c]);
            fp16_l2 += d1 * d1;
            pos16_l2 += d2 * d2;
            fp16_max = max(fp16_max, d1);
            pos16_max = max(pos16_max, d2);
        }
        csv << n << "," << sqrt(fp16_l2) << "," << sqrt(pos16_l2)
            << "," << fp16_max << "," << pos16_max << "\n";
    }
}

int main() {
    cout << "=========================================================================\n";
    cout << "  2-LAYER GCN NUMERICAL ACCURACY (Initial Weights, Seed 2026)            \n";
    cout << "  Pure sw::universal — Long Double Ground Truth                           \n";
    cout << "=========================================================================\n\n";

    auto xr = load_binary<float>("data/features_fp32.bin", N * F);
    auto er = load_binary<float>("data/norm_edge_weight_fp32.bin", E);
    auto edges = load_binary<int64_t>("data/norm_edge_index_int64.bin", 2 * E);
    auto raw_w = load_binary<float>("checkpoints/gcn_initial_weights_2layer_seed2026.bin", H * F + H + C * H + C);
    auto labels = load_binary<int64_t>("data/labels_int64.bin", N);
    auto test_mask = load_binary<uint8_t>("data/test_mask_uint8.bin", N);

    cout << "[1] Computing Long Double ground truth...\n";
    auto logits_ld = run_2layer_forward<LongDouble>(xr, er, edges, raw_w);

    cout << "[2] Computing Universal FP32 baseline...\n";
    auto logits_fp32 = run_2layer_forward<FP32_Uni>(xr, er, edges, raw_w);

    cout << "[3] Computing Universal FP16 (cfloat<16,5>)...\n";
    auto logits_fp16 = run_2layer_forward<FP16_Uni>(xr, er, edges, raw_w);

    cout << "[4] Computing Universal Posit16 (posit<16,1>)...\n";
    auto logits_pos16 = run_2layer_forward<Posit16_Uni>(xr, er, edges, raw_w);

    // Error vs Long Double
    auto err_fp32  = compute_errors(logits_fp32,  logits_ld);
    auto err_fp16  = compute_errors(logits_fp16,  logits_ld);
    auto err_pos16 = compute_errors(logits_pos16, logits_ld);

    cout << "\n========================================================================================\n";
    cout << "          NUMERICAL ERROR vs LONG DOUBLE GROUND TRUTH (2-Layer Forward Pass)             \n";
    cout << "========================================================================================\n";
    cout << left << setw(20) << "Format"
         << setw(18) << "MAE"
         << setw(18) << "RMSE"
         << setw(18) << "Max Abs Error"
         << setw(18) << "Median Error" << "\n";
    cout << "----------------------------------------------------------------------------------------\n";

    auto prt = [](const string& name, const ErrorMetrics& e) {
        cout << left << setw(20) << name
             << scientific << setprecision(6)
             << setw(18) << e.mae
             << setw(18) << e.rmse
             << setw(18) << e.max_err
             << setw(18) << e.median_err << "\n";
    };
    prt("Universal FP32", err_fp32);
    prt("Universal FP16", err_fp16);
    prt("Universal Posit16", err_pos16);
    cout << "========================================================================================\n";

    // Ratios
    cout << "\n  FP16/Posit16 Error Ratios:\n";
    cout << "    MAE Ratio:       " << fixed << setprecision(2) << err_fp16.mae / err_pos16.mae << "x\n";
    cout << "    RMSE Ratio:      " << err_fp16.rmse / err_pos16.rmse << "x\n";
    cout << "    Max Error Ratio: " << err_fp16.max_err / err_pos16.max_err << "x\n";
    cout << "    Median Ratio:    " << err_fp16.median_err / err_pos16.median_err << "x\n";

    // Classification accuracy with each format
    cout << "\n========================================================================================\n";
    cout << "          CLASSIFICATION ACCURACY (Initial Weights, Test Set, 1000 nodes)                \n";
    cout << "========================================================================================\n";

    auto eval_acc = [&](const string& name, const vector<double>& logits) {
        int correct = 0, count = 0;
        for (int n = 0; n < N; ++n) {
            if (!test_mask[n]) continue;
            int pred = 0;
            double best = logits[n * C];
            for (int c = 1; c < C; ++c) {
                if (logits[n * C + c] > best) { best = logits[n * C + c]; pred = c; }
            }
            if (pred == static_cast<int>(labels[n])) ++correct;
            ++count;
        }
        cout << "  " << left << setw(20) << name << ": " << fixed << setprecision(2)
             << 100.0 * correct / count << "% (" << correct << "/" << count << ")\n";
    };

    eval_acc("Long Double (Ref)", logits_ld);
    eval_acc("Universal FP32",    logits_fp32);
    eval_acc("Universal FP16",    logits_fp16);
    eval_acc("Universal Posit16", logits_pos16);
    cout << "========================================================================================\n";

    // Agreement check: how many nodes' predictions differ from long double?
    int fp16_disagree = 0, pos16_disagree = 0;
    for (int n = 0; n < N; ++n) {
        if (!test_mask[n]) continue;
        int ref_pred = 0, fp16_pred = 0, pos16_pred = 0;
        double ref_best = logits_ld[n * C], fp16_best = logits_fp16[n * C], pos16_best = logits_pos16[n * C];
        for (int c = 1; c < C; ++c) {
            if (logits_ld[n * C + c] > ref_best) { ref_best = logits_ld[n * C + c]; ref_pred = c; }
            if (logits_fp16[n * C + c] > fp16_best) { fp16_best = logits_fp16[n * C + c]; fp16_pred = c; }
            if (logits_pos16[n * C + c] > pos16_best) { pos16_best = logits_pos16[n * C + c]; pos16_pred = c; }
        }
        if (fp16_pred != ref_pred) ++fp16_disagree;
        if (pos16_pred != ref_pred) ++pos16_disagree;
    }

    cout << "\n  Prediction Disagreements vs Long Double (Test Set):\n";
    cout << "    FP16:    " << fp16_disagree << " nodes disagree\n";
    cout << "    Posit16: " << pos16_disagree << " nodes disagree\n";

    // Save per-node errors
    save_per_node_errors("results/logs/per_node_numerical_error.csv", logits_fp16, logits_pos16, logits_ld);
    cout << "\n  Saved per-node error CSV: results/logs/per_node_numerical_error.csv\n";

    // Save summary CSV
    ofstream summary("results/logs/numerical_accuracy_summary.csv");
    summary << "format,mae,rmse,max_error,median_error\n";
    summary << "FP32," << scientific << setprecision(8) << err_fp32.mae << "," << err_fp32.rmse << "," << err_fp32.max_err << "," << err_fp32.median_err << "\n";
    summary << "FP16," << err_fp16.mae << "," << err_fp16.rmse << "," << err_fp16.max_err << "," << err_fp16.median_err << "\n";
    summary << "Posit16," << err_pos16.mae << "," << err_pos16.rmse << "," << err_pos16.max_err << "," << err_pos16.median_err << "\n";
    cout << "  Saved summary CSV: results/logs/numerical_accuracy_summary.csv\n";

    return 0;
}

