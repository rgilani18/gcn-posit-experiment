#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// Pure Stillwater Universal (NO PyTorch)
#include <universal/number/cfloat/cfloat.hpp>
#include <universal/number/posit/posit.hpp>

using namespace std;

using FP16_Type    = sw::universal::cfloat<16, 5>;
using Posit16_Type = sw::universal::posit<16, 1>;

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
        cerr << "Error opening " << filename << '\n';
        exit(1);
    }
    file.read(reinterpret_cast<char*>(data.data()), count * sizeof(T));
    return data;
}

// -----------------------------------------------------------------------------
// Unified 2-Layer GCN Engine: Regime 3 (Double 64-Bit Adam Optimizer States)
// -----------------------------------------------------------------------------
template <typename T>
void run_2layer_double_adam(
    const string& format_name,
    const string& log_filename,
    const vector<float>& xr,
    const vector<float>& er,
    const vector<int64_t>& edges,
    const vector<int64_t>& labels,
    const vector<uint8_t>& train_mask,
    const vector<uint8_t>& val_mask,
    const vector<uint8_t>& test_mask,
    const vector<float>& raw_w,
    int epochs = 200,
    double lr_val = 0.01,
    double dropout_rate = 0.5,
    double wd_val = 5e-4
) {
    cout << "\n=========================================================================\n";
    cout << "  REGIME 3 (DOUBLE 64-BIT ADAM STATES): " << format_name << " (200 Epochs)\n";
    cout << "=========================================================================\n";

    const double beta1_d = 0.9;
    const double beta2_d = 0.999;
    const double eps_d = 1e-4;
    const double lr_d = lr_val;
    const double wd_d = wd_val;
    const T drop_scale = T(1.0 / (1.0 - dropout_rate));

    vector<T> x(N * F), ew(E);
    for (size_t i = 0; i < x.size(); ++i) x[i] = T(xr[i]);
    for (int e = 0; e < E; ++e) ew[e] = T(er[e]);

    vector<T> w1(H * F), b1(H);
    vector<T> w2(C * H), b2(C);
    size_t offset = 0;
    for (size_t i = 0; i < w1.size(); ++i) w1[i] = T(raw_w[offset++]);
    for (size_t i = 0; i < b1.size(); ++i) b1[i] = T(raw_w[offset++]);
    for (size_t i = 0; i < w2.size(); ++i) w2[i] = T(raw_w[offset++]);
    for (size_t i = 0; i < b2.size(); ++i) b2[i] = T(raw_w[offset++]);

    // Adam First and Second Moments Stored in 64-bit double
    vector<double> m_w1(w1.size(), 0.0), v_w1(w1.size(), 0.0);
    vector<double> m_b1(b1.size(), 0.0), v_b1(b1.size(), 0.0);
    vector<double> m_w2(w2.size(), 0.0), v_w2(w2.size(), 0.0);
    vector<double> m_b2(b2.size(), 0.0), v_b2(b2.size(), 0.0);

    int train_count = 0, val_count = 0, test_count = 0;
    for (int n = 0; n < N; ++n) {
        if (train_mask[n]) ++train_count;
        if (val_mask[n])   ++val_count;
        if (test_mask[n])  ++test_count;
    }

    ofstream csv(log_filename);
    csv << "epoch,train_loss,train_acc,val_loss,val_acc,test_loss,test_acc\n";

    double best_val_loss = INFINITY;
    double best_val_acc = 0.0;
    double corresponding_test_acc = 0.0;
    int best_epoch = 0;

    mt19937 rng(42);
    uniform_real_distribution<double> uniform_dist(0.0, 1.0);

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        // Forward pass with Dropout
        vector<T> s1(N * H, T(0));
        for (int n = 0; n < N; ++n)
            for (int h = 0; h < H; ++h)
                for (int f = 0; f < F; ++f)
                    s1[n * H + h] += x[n * F + f] * w1[h * F + f];

        vector<T> agg1(N * H, T(0));
        for (int e = 0; e < E; ++e) {
            int src = static_cast<int>(edges[e]), dst = static_cast<int>(edges[E + e]);
            for (int h = 0; h < H; ++h) agg1[dst * H + h] += ew[e] * s1[src * H + h];
        }

        vector<T> pre_relu(N * H), h_act(N * H);
        for (int n = 0; n < N; ++n) {
            for (int h = 0; h < H; ++h) {
                T v = agg1[n * H + h] + b1[h];
                pre_relu[n * H + h] = v;
                h_act[n * H + h] = v > T(0) ? v : T(0);
            }
        }

        vector<uint8_t> drop_mask(N * H, 1);
        vector<T> h_dropped(N * H);
        for (size_t i = 0; i < h_dropped.size(); ++i) {
            if (uniform_dist(rng) < dropout_rate) {
                drop_mask[i] = 0;
                h_dropped[i] = T(0);
            } else {
                drop_mask[i] = 1;
                h_dropped[i] = h_act[i] * drop_scale;
            }
        }

        vector<T> s2(N * C, T(0));
        for (int n = 0; n < N; ++n)
            for (int c = 0; c < C; ++c)
                for (int h = 0; h < H; ++h)
                    s2[n * C + c] += h_dropped[n * H + h] * w2[c * H + h];

        vector<T> logits(N * C, T(0));
        for (int e = 0; e < E; ++e) {
            int src = static_cast<int>(edges[e]), dst = static_cast<int>(edges[E + e]);
            for (int c = 0; c < C; ++c) logits[dst * C + c] += ew[e] * s2[src * C + c];
        }
        for (int n = 0; n < N; ++n)
            for (int c = 0; c < C; ++c) logits[n * C + c] += b2[c];

        vector<T> probs(N * C);
        double train_loss = 0.0;
        int train_correct = 0;
        for (int n = 0; n < N; ++n) {
            double max_val = -INFINITY;
            for (int c = 0; c < C; ++c) {
                double val = static_cast<double>(logits[n * C + c]);
                if (val > max_val) max_val = val;
            }
            double sum_exp = 0.0;
            for (int c = 0; c < C; ++c) sum_exp += exp(static_cast<double>(logits[n * C + c]) - max_val);
            int pred_class = 0;
            double max_prob = -1.0;
            for (int c = 0; c < C; ++c) {
                double p = exp(static_cast<double>(logits[n * C + c]) - max_val) / sum_exp;
                probs[n * C + c] = T(p);
                if (p > max_prob) { max_prob = p; pred_class = c; }
            }
            if (train_mask[n]) {
                int y = static_cast<int>(labels[n]);
                train_loss -= log(max(static_cast<double>(probs[n * C + y]), 1e-15));
                if (pred_class == y) ++train_correct;
            }
        }
        train_loss /= train_count;
        double train_acc = 100.0 * train_correct / train_count;

        // Backward Pass
        vector<T> dlogits(N * C, T(0));
        for (int n = 0; n < N; ++n) {
            if (!train_mask[n]) continue;
            int y = static_cast<int>(labels[n]);
            for (int c = 0; c < C; ++c) {
                double target = (c == y) ? 1.0 : 0.0;
                dlogits[n * C + c] = (probs[n * C + c] - T(target)) / T(train_count);
            }
        }

        vector<T> dS2(N * C, T(0));
        vector<T> db2(C, T(0));
        for (int e = 0; e < E; ++e) {
            int src = static_cast<int>(edges[e]), dst = static_cast<int>(edges[E + e]);
            for (int c = 0; c < C; ++c) dS2[src * C + c] += ew[e] * dlogits[dst * C + c];
        }
        for (int n = 0; n < N; ++n) {
            if (!train_mask[n]) continue;
            for (int c = 0; c < C; ++c) db2[c] += dlogits[n * C + c];
        }

        vector<T> dW2(C * H, T(0));
        for (int n = 0; n < N; ++n)
            for (int c = 0; c < C; ++c)
                for (int h = 0; h < H; ++h)
                    dW2[c * H + h] += dS2[n * C + c] * h_dropped[n * H + h];

        vector<T> dH_dropped(N * H, T(0));
        for (int n = 0; n < N; ++n)
            for (int h = 0; h < H; ++h)
                for (int c = 0; c < C; ++c)
                    dH_dropped[n * H + h] += dS2[n * C + c] * w2[c * H + h];

        vector<T> dH_act(N * H);
        for (size_t i = 0; i < dH_act.size(); ++i) {
            dH_act[i] = drop_mask[i] ? (dH_dropped[i] * drop_scale) : T(0);
        }

        vector<T> dZ1(N * H);
        for (size_t i = 0; i < dZ1.size(); ++i) {
            dZ1[i] = pre_relu[i] > T(0) ? dH_act[i] : T(0);
        }

        vector<T> dS1(N * H, T(0));
        vector<T> db1(H, T(0));
        for (int e = 0; e < E; ++e) {
            int src = static_cast<int>(edges[e]), dst = static_cast<int>(edges[E + e]);
            for (int h = 0; h < H; ++h) dS1[src * H + h] += ew[e] * dZ1[dst * H + h];
        }
        for (int n = 0; n < N; ++n) {
            if (!train_mask[n]) continue;
            for (int h = 0; h < H; ++h) db1[h] += dZ1[n * H + h];
        }

        vector<T> dW1(H * F, T(0));
        for (int n = 0; n < N; ++n)
            for (int h = 0; h < H; ++h)
                for (int f = 0; f < F; ++f)
                    dW1[h * F + f] += dS1[n * H + h] * x[n * F + f];

        // Adam Step (Double 64-bit moments)
        double bias_correction1 = 1.0 - std::pow(beta1_d, epoch);
        double bias_correction2 = 1.0 - std::pow(beta2_d, epoch);

        auto adam_step_double = [&](vector<T>& param, const vector<T>& grad,
                                    vector<double>& m, vector<double>& v, bool apply_wd) {
            for (size_t i = 0; i < param.size(); ++i) {
                double g = static_cast<double>(grad[i]);
                double p = static_cast<double>(param[i]);
                if (apply_wd) g += wd_d * p;

                m[i] = beta1_d * m[i] + (1.0 - beta1_d) * g;
                v[i] = beta2_d * v[i] + (1.0 - beta2_d) * g * g;

                double m_hat = m[i] / bias_correction1;
                double v_hat = v[i] / bias_correction2;
                double denom = std::sqrt(std::max(v_hat, 0.0)) + eps_d;

                double step = lr_d * m_hat / denom;
                param[i] = T(p - step);
            }
        };

        adam_step_double(w1, dW1, m_w1, v_w1, true);   // Apply weight decay on W1
        adam_step_double(b1, db1, m_b1, v_b1, false);
        adam_step_double(w2, dW2, m_w2, v_w2, false);
        adam_step_double(b2, db2, m_b2, v_b2, false);

        // Deterministic Evaluation (Dropout OFF)
        auto evaluate_split = [&](const vector<uint8_t>& mask, int count, double& out_loss, double& out_acc) {
            vector<T> vs1(N * H, T(0));
            for (int n = 0; n < N; ++n)
                for (int h = 0; h < H; ++h)
                    for (int f = 0; f < F; ++f)
                        vs1[n * H + h] += x[n * F + f] * w1[h * F + f];

            vector<T> vagg1(N * H, T(0));
            for (int e = 0; e < E; ++e) {
                int src = static_cast<int>(edges[e]), dst = static_cast<int>(edges[E + e]);
                for (int h = 0; h < H; ++h) vagg1[dst * H + h] += ew[e] * vs1[src * H + h];
            }
            vector<T> vh(N * H);
            for (int n = 0; n < N; ++n) {
                for (int h = 0; h < H; ++h) {
                    T v = vagg1[n * H + h] + b1[h];
                    vh[n * H + h] = v > T(0) ? v : T(0);
                }
            }

            vector<T> vs2(N * C, T(0));
            for (int n = 0; n < N; ++n)
                for (int c = 0; c < C; ++c)
                    for (int h = 0; h < H; ++h)
                        vs2[n * C + c] += vh[n * H + h] * w2[c * H + h];

            vector<T> vlogits(N * C, T(0));
            for (int e = 0; e < E; ++e) {
                int src = static_cast<int>(edges[e]), dst = static_cast<int>(edges[E + e]);
                for (int c = 0; c < C; ++c) vlogits[dst * C + c] += ew[e] * vs2[src * C + c];
            }
            for (int n = 0; n < N; ++n)
                for (int c = 0; c < C; ++c) vlogits[n * C + c] += b2[c];

            out_loss = 0.0;
            int correct = 0;
            for (int n = 0; n < N; ++n) {
                if (!mask[n]) continue;
                double max_val = -INFINITY;
                for (int c = 0; c < C; ++c) {
                    double val = static_cast<double>(vlogits[n * C + c]);
                    if (val > max_val) max_val = val;
                }
                double sum_exp = 0.0;
                for (int c = 0; c < C; ++c) sum_exp += exp(static_cast<double>(vlogits[n * C + c]) - max_val);
                int pred_class = 0;
                double max_prob = -1.0;
                for (int c = 0; c < C; ++c) {
                    double p = exp(static_cast<double>(vlogits[n * C + c]) - max_val) / sum_exp;
                    if (p > max_prob) { max_prob = p; pred_class = c; }
                }
                int y = static_cast<int>(labels[n]);
                double py = exp(static_cast<double>(vlogits[n * C + y]) - max_val) / sum_exp;
                out_loss -= log(max(py, 1e-15));
                if (pred_class == y) ++correct;
            }
            out_loss /= count;
            out_acc = 100.0 * correct / count;
        };

        double val_loss = 0.0, val_acc = 0.0;
        double test_loss = 0.0, test_acc = 0.0;
        evaluate_split(val_mask, val_count, val_loss, val_acc);
        evaluate_split(test_mask, test_count, test_loss, test_acc);

        if (val_loss < best_val_loss) {
            best_val_loss = val_loss;
            best_val_acc = val_acc;
            corresponding_test_acc = test_acc;
            best_epoch = epoch;
        }

        csv << epoch << "," << train_loss << "," << train_acc << ","
            << val_loss << "," << val_acc << "," << test_loss << "," << test_acc << "\n";

        if (epoch % 20 == 0 || epoch == 1 || epoch == epochs) {
            cout << "Epoch " << setw(3) << epoch << " | "
                 << "TrL: " << fixed << setprecision(4) << train_loss << " TrAcc: " << setprecision(1) << train_acc << "% | "
                 << "ValL: " << setprecision(4) << val_loss << " ValAcc: " << setprecision(1) << val_acc << "% | "
                 << "TestL: " << setprecision(4) << test_loss << " TestAcc: " << setprecision(1) << test_acc << "%\n";
        }
    }
    cout << "Finished " << format_name << ". Best Val Epoch: " << best_epoch 
         << " (Val Loss: " << best_val_loss << ", Val Acc: " << best_val_acc 
         << "%, Test Acc: " << corresponding_test_acc << "%)\n";
}

int main(int argc, char* argv[]) {
    int epochs = 200;
    if (argc > 1) epochs = atoi(argv[1]);

    auto xr = load_binary<float>("data/features_fp32.bin", N * F);
    auto er = load_binary<float>("data/norm_edge_weight_fp32.bin", E);
    auto edges = load_binary<int64_t>("data/norm_edge_index_int64.bin", 2 * E);
    auto labels = load_binary<int64_t>("data/labels_int64.bin", N);
    auto train_mask = load_binary<uint8_t>("data/train_mask_uint8.bin", N);
    auto val_mask = load_binary<uint8_t>("data/val_mask_uint8.bin", N);
    auto test_mask = load_binary<uint8_t>("data/test_mask_uint8.bin", N);
    auto raw_w = load_binary<float>("checkpoints/gcn_initial_weights_2layer_seed2026.bin", H * F + H + C * H + C);

    // 1. Run FP16 with 64-bit Double Adam States
    run_2layer_double_adam<FP16_Type>(
        "FP16", "results/logs/training_log_200ep_fp16_adam_double.csv",
        xr, er, edges, labels, train_mask, val_mask, test_mask, raw_w, epochs
    );

    // 2. Run Posit16 with 64-bit Double Adam States
    run_2layer_double_adam<Posit16_Type>(
        "Posit16", "results/logs/training_log_200ep_posit16_adam_double.csv",
        xr, er, edges, labels, train_mask, val_mask, test_mask, raw_w, epochs
    );

    return 0;
}

