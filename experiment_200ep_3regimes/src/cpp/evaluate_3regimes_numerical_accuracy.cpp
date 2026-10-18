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
    if (!file) { cerr << "Error opening " << filename << '\n'; exit(1); }
    file.read(reinterpret_cast<char*>(data.data()), count * sizeof(T));
    return data;
}

template <typename T>
void save_weights_binary(const string& filename,
                         const vector<T>& w1, const vector<T>& b1,
                         const vector<T>& w2, const vector<T>& b2) {
    ofstream file(filename, ios::binary);
    if (!file) { cerr << "Error writing " << filename << '\n'; return; }
    auto write_vec = [&](const vector<T>& v) {
        for (const auto& val : v) {
            float f = static_cast<float>(static_cast<double>(val));
            file.write(reinterpret_cast<const char*>(&f), sizeof(float));
        }
    };
    write_vec(w1); write_vec(b1); write_vec(w2); write_vec(b2);
}

struct NumericalMetrics {
    double test_acc;
    double val_loss;
    double mae;
    double rmse;
    double max_err;
    int disagreements;
};

// Evaluates trained weights in type T vs 80-bit Long Double ground truth
template <typename T>
NumericalMetrics audit_numerical_accuracy(
    const vector<float>& xr, const vector<float>& er, const vector<int64_t>& edges,
    const vector<int64_t>& labels, const vector<uint8_t>& test_mask,
    const vector<T>& w1_t, const vector<T>& b1_t,
    const vector<T>& w2_t, const vector<T>& b2_t,
    double val_loss_val
) {
    // 1. Forward Pass in 16-bit (Type T)
    vector<T> x(N * F), ew(E);
    for (size_t i = 0; i < x.size(); ++i) x[i] = T(xr[i]);
    for (int e = 0; e < E; ++e) ew[e] = T(er[e]);

    vector<T> s1_t(N * H, T(0));
    for (int n = 0; n < N; ++n)
        for (int h = 0; h < H; ++h)
            for (int f = 0; f < F; ++f)
                s1_t[n * H + h] += x[n * F + f] * w1_t[h * F + f];

    vector<T> agg1_t(N * H, T(0));
    for (int e = 0; e < E; ++e) {
        int src = static_cast<int>(edges[e]), dst = static_cast<int>(edges[E + e]);
        for (int h = 0; h < H; ++h) agg1_t[dst * H + h] += ew[e] * s1_t[src * H + h];
    }

    vector<T> h_act_t(N * H);
    for (int n = 0; n < N; ++n)
        for (int h = 0; h < H; ++h) {
            T v = agg1_t[n * H + h] + b1_t[h];
            h_act_t[n * H + h] = v > T(0) ? v : T(0);
        }

    vector<T> s2_t(N * C, T(0));
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c)
            for (int h = 0; h < H; ++h)
                s2_t[n * C + c] += h_act_t[n * H + h] * w2_t[c * H + h];

    vector<T> logits_t(N * C, T(0));
    for (int e = 0; e < E; ++e) {
        int src = static_cast<int>(edges[e]), dst = static_cast<int>(edges[E + e]);
        for (int c = 0; c < C; ++c) logits_t[dst * C + c] += ew[e] * s2_t[src * C + c];
    }
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c) logits_t[n * C + c] += b2_t[c];

    // 2. Exact same trained weights evaluated in 80-bit Long Double Ground Truth
    vector<long double> x_ld(N * F), ew_ld(E);
    for (size_t i = 0; i < x_ld.size(); ++i) x_ld[i] = static_cast<long double>(xr[i]);
    for (int e = 0; e < E; ++e) ew_ld[e] = static_cast<long double>(er[e]);

    vector<long double> w1_ld(w1_t.size()), b1_ld(b1_t.size()), w2_ld(w2_t.size()), b2_ld(b2_t.size());
    for (size_t i = 0; i < w1_t.size(); ++i) w1_ld[i] = static_cast<long double>(static_cast<double>(w1_t[i]));
    for (size_t i = 0; i < b1_t.size(); ++i) b1_ld[i] = static_cast<long double>(static_cast<double>(b1_t[i]));
    for (size_t i = 0; i < w2_t.size(); ++i) w2_ld[i] = static_cast<long double>(static_cast<double>(w2_t[i]));
    for (size_t i = 0; i < b2_t.size(); ++i) b2_ld[i] = static_cast<long double>(static_cast<double>(b2_t[i]));

    vector<long double> s1_ld(N * H, 0.0L);
    for (int n = 0; n < N; ++n)
        for (int h = 0; h < H; ++h)
            for (int f = 0; f < F; ++f)
                s1_ld[n * H + h] += x_ld[n * F + f] * w1_ld[h * F + f];

    vector<long double> agg1_ld(N * H, 0.0L);
    for (int e = 0; e < E; ++e) {
        int src = static_cast<int>(edges[e]), dst = static_cast<int>(edges[E + e]);
        for (int h = 0; h < H; ++h) agg1_ld[dst * H + h] += ew_ld[e] * s1_ld[src * H + h];
    }

    vector<long double> h_act_ld(N * H);
    for (int n = 0; n < N; ++n)
        for (int h = 0; h < H; ++h) {
            long double v = agg1_ld[n * H + h] + b1_ld[h];
            h_act_ld[n * H + h] = v > 0.0L ? v : 0.0L;
        }

    vector<long double> s2_ld(N * C, 0.0L);
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c)
            for (int h = 0; h < H; ++h)
                s2_ld[n * C + c] += h_act_ld[n * H + h] * w2_ld[c * H + h];

    vector<long double> logits_ld(N * C, 0.0L);
    for (int e = 0; e < E; ++e) {
        int src = static_cast<int>(edges[e]), dst = static_cast<int>(edges[E + e]);
        for (int c = 0; c < C; ++c) logits_ld[dst * C + c] += ew_ld[e] * s2_ld[src * C + c];
    }
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c) logits_ld[n * C + c] += b2_ld[c];

    // 3. Compare Logits: MAE, RMSE, Max Absolute Error
    double sum_abs = 0.0, sum_sq = 0.0, max_err = 0.0;
    size_t count = logits_t.size();
    for (size_t i = 0; i < count; ++i) {
        double diff = fabs(static_cast<double>(logits_t[i]) - static_cast<double>(logits_ld[i]));
        sum_abs += diff;
        sum_sq += diff * diff;
        max_err = max(max_err, diff);
    }
    double mae = sum_abs / count;
    double rmse = sqrt(sum_sq / count);

    // 4. Test Accuracy & Disagreements vs Ground Truth
    int test_count = 0, correct = 0, disagreements = 0;
    for (int n = 0; n < N; ++n) {
        if (!test_mask[n]) continue;
        int pred_t = 0, pred_ld = 0;
        double max_t = -1e30, max_ld = -1e30;
        for (int c = 0; c < C; ++c) {
            double z_t = static_cast<double>(logits_t[n * C + c]);
            double z_ld = static_cast<double>(logits_ld[n * C + c]);
            if (z_t > max_t)   { max_t = z_t; pred_t = c; }
            if (z_ld > max_ld) { max_ld = z_ld; pred_ld = c; }
        }
        if (pred_t == static_cast<int>(labels[n])) ++correct;
        if (pred_t != pred_ld) ++disagreements;
        ++test_count;
    }
    double test_acc = 100.0 * correct / test_count;

    return {test_acc, val_loss_val, mae, rmse, max_err, disagreements};
}

// =============================================================================
// Unified Runner for a Given Regime
// =============================================================================
template <typename T>
NumericalMetrics train_and_evaluate_regime(
    const string& regime_name, const string& format_name,
    const vector<float>& xr, const vector<float>& er, const vector<int64_t>& edges,
    const vector<int64_t>& labels,
    const vector<uint8_t>& train_mask, const vector<uint8_t>& val_mask, const vector<uint8_t>& test_mask,
    const vector<float>& raw_w,
    int regime_id, // 1=Vanilla GD, 2=Pure 16-bit Adam, 3=Double Adam
    int epochs = 200
) {
    cout << "  -> Running " << format_name << " (" << epochs << " epochs)..." << flush;

    double lr_val = (regime_id == 1) ? 0.5 : 0.01;
    double dropout_rate = 0.5, wd_val = 5e-4;
    const T lr = T(lr_val);
    const T drop_scale = T(1.0 / (1.0 - dropout_rate));
    const T weight_decay = T(wd_val);

    const T beta1_t = T(0.9), beta2_t = T(0.999), adam_eps_t = T(1e-4);
    const double beta1_d = 0.9, beta2_d = 0.999, eps_d = 1e-4, lr_d = 0.01, wd_d = 5e-4;

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

    vector<T> m_w1_t, v_w1_t, m_b1_t, v_b1_t, m_w2_t, v_w2_t, m_b2_t, v_b2_t;
    vector<double> m_w1_d, v_w1_d, m_b1_d, v_b1_d, m_w2_d, v_w2_d, m_b2_d, v_b2_d;
    if (regime_id == 2) {
        m_w1_t.assign(w1.size(), T(0)); v_w1_t.assign(w1.size(), T(0));
        m_b1_t.assign(b1.size(), T(0)); v_b1_t.assign(b1.size(), T(0));
        m_w2_t.assign(w2.size(), T(0)); v_w2_t.assign(w2.size(), T(0));
        m_b2_t.assign(b2.size(), T(0)); v_b2_t.assign(b2.size(), T(0));
    } else if (regime_id == 3) {
        m_w1_d.assign(w1.size(), 0.0); v_w1_d.assign(w1.size(), 0.0);
        m_b1_d.assign(b1.size(), 0.0); v_b1_d.assign(b1.size(), 0.0);
        m_w2_d.assign(w2.size(), 0.0); v_w2_d.assign(w2.size(), 0.0);
        m_b2_d.assign(b2.size(), 0.0); v_b2_d.assign(b2.size(), 0.0);
    }

    int train_count = 0, val_count = 0;
    for (int n = 0; n < N; ++n) {
        if (train_mask[n]) ++train_count;
        if (val_mask[n])   ++val_count;
    }

    mt19937 rng(42);
    uniform_real_distribution<double> uniform_dist(0.0, 1.0);
    double last_val_loss = 0.0;

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        // Forward Pass (Dropout enabled)
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
        for (int n = 0; n < N; ++n) {
            double max_val = -INFINITY;
            for (int c = 0; c < C; ++c) {
                double val = static_cast<double>(logits[n * C + c]);
                if (val > max_val) max_val = val;
            }
            double sum_exp = 0.0;
            for (int c = 0; c < C; ++c) sum_exp += exp(static_cast<double>(logits[n * C + c]) - max_val);
            for (int c = 0; c < C; ++c) {
                double p = exp(static_cast<double>(logits[n * C + c]) - max_val) / sum_exp;
                probs[n * C + c] = T(p);
            }
        }

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

        // Parameter Updates
        if (regime_id == 1) {
            for (size_t i = 0; i < w1.size(); ++i) w1[i] = w1[i] - lr * (dW1[i] + weight_decay * w1[i]);
            for (size_t i = 0; i < b1.size(); ++i) b1[i] = b1[i] - lr * db1[i];
            for (size_t i = 0; i < w2.size(); ++i) w2[i] = w2[i] - lr * dW2[i];
            for (size_t i = 0; i < b2.size(); ++i) b2[i] = b2[i] - lr * db2[i];
        } else if (regime_id == 2) {
            double bc1 = 1.0 - pow(0.9, epoch), bc2 = 1.0 - pow(0.999, epoch);
            T bias_corr1 = T(bc1), bias_corr2 = T(bc2);
            auto adam_step = [&](vector<T>& param, const vector<T>& grad,
                                 vector<T>& m, vector<T>& v, bool apply_wd) {
                for (size_t i = 0; i < param.size(); ++i) {
                    T g = grad[i];
                    if (apply_wd) g += weight_decay * param[i];
                    m[i] = beta1_t * m[i] + (T(1) - beta1_t) * g;
                    v[i] = beta2_t * v[i] + (T(1) - beta2_t) * g * g;
                    T m_hat = m[i] / bias_corr1;
                    T v_hat = v[i] / bias_corr2;
                    double v_hat_val = max(static_cast<double>(v_hat), 0.0);
                    T denom = T(sqrt(v_hat_val)) + adam_eps_t;
                    param[i] = param[i] - lr * (m_hat / denom);
                }
            };
            adam_step(w1, dW1, m_w1_t, v_w1_t, true);
            adam_step(b1, db1, m_b1_t, v_b1_t, false);
            adam_step(w2, dW2, m_w2_t, v_w2_t, false);
            adam_step(b2, db2, m_b2_t, v_b2_t, false);
        } else if (regime_id == 3) {
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
            adam_step_double(w1, dW1, m_w1_d, v_w1_d, true);
            adam_step_double(b1, db1, m_b1_d, v_b1_d, false);
            adam_step_double(w2, dW2, m_w2_d, v_w2_d, false);
            adam_step_double(b2, db2, m_b2_d, v_b2_d, false);
        }

        if (epoch == epochs) {
            // Compute final val loss
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
            for (int n = 0; n < N; ++n)
                for (int h = 0; h < H; ++h) {
                    T v = vagg1[n * H + h] + b1[h];
                    vh[n * H + h] = v > T(0) ? v : T(0);
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

            double vl = 0.0;
            for (int n = 0; n < N; ++n) {
                if (!val_mask[n]) continue;
                double max_val = -INFINITY;
                for (int c = 0; c < C; ++c) max_val = max(max_val, static_cast<double>(vlogits[n * C + c]));
                double sum_exp = 0.0;
                for (int c = 0; c < C; ++c) sum_exp += exp(static_cast<double>(vlogits[n * C + c]) - max_val);
                int y = static_cast<int>(labels[n]);
                double py = exp(static_cast<double>(vlogits[n * C + y]) - max_val) / sum_exp;
                vl -= log(max(py, 1e-15));
            }
            last_val_loss = vl / val_count;
        }
    }

    // Numerical Audit at epoch 200
    auto metrics = audit_numerical_accuracy<T>(
        xr, er, edges, labels, test_mask, w1, b1, w2, b2, last_val_loss
    );

    cout << " Done! Acc: " << fixed << setprecision(1) << metrics.test_acc << "%"
         << " | MAE: " << scientific << setprecision(3) << metrics.mae
         << " | MaxErr: " << metrics.max_err << endl;

    return metrics;
}

int main() {
    cout << "=========================================================================\n";
    cout << "  COMPREHENSIVE NUMERICAL ACCURACY AUDIT ACROSS ALL 3 REGIMES (200 Ep)   \n";
    cout << "  FP16 vs Posit16 vs Long Double Ground Truth (Cora Dataset, Seed 2026)  \n";
    cout << "=========================================================================\n\n";

    auto xr = load_binary<float>("data/features_fp32.bin", N * F);
    auto er = load_binary<float>("data/norm_edge_weight_fp32.bin", E);
    auto edges = load_binary<int64_t>("data/norm_edge_index_int64.bin", 2 * E);
    auto labels = load_binary<int64_t>("data/labels_int64.bin", N);
    auto train_mask = load_binary<uint8_t>("data/train_mask_uint8.bin", N);
    auto val_mask = load_binary<uint8_t>("data/val_mask_uint8.bin", N);
    auto test_mask = load_binary<uint8_t>("data/test_mask_uint8.bin", N);
    auto raw_w = load_binary<float>("checkpoints/gcn_initial_weights_2layer_seed2026.bin", H * F + H + C * H + C);

    ofstream csv("results/logs/numerical_accuracy_3regimes.csv");
    csv << "regime,format,test_acc,val_loss,mae,rmse,max_err,disagreements\n";

    auto record = [&](const string& regime, const string& format, const NumericalMetrics& m) {
        csv << regime << "," << format << ","
            << fixed << setprecision(2) << m.test_acc << ","
            << setprecision(4) << m.val_loss << ","
            << scientific << setprecision(8)
            << m.mae << "," << m.rmse << "," << m.max_err << ","
            << m.disagreements << "\n";
    };

    // 1. Regime 1: Vanilla GD
    cout << "\n[Regime 1: Without Optimizer (Vanilla GD, 200 Epochs)]\n";
    auto r1_fp16 = train_and_evaluate_regime<FP16_Type>("Regime 1", "FP16", xr, er, edges, labels, train_mask, val_mask, test_mask, raw_w, 1);
    auto r1_pos  = train_and_evaluate_regime<Posit16_Type>("Regime 1", "Posit16", xr, er, edges, labels, train_mask, val_mask, test_mask, raw_w, 1);
    record("Regime 1 (Vanilla GD)", "FP16", r1_fp16);
    record("Regime 1 (Vanilla GD)", "Posit16", r1_pos);

    // 2. Regime 2: Pure 16-bit Adam
    cout << "\n[Regime 2: Pure 16-Bit Optimizer (16-bit Adam Moments, 200 Epochs)]\n";
    auto r2_fp16 = train_and_evaluate_regime<FP16_Type>("Regime 2", "FP16", xr, er, edges, labels, train_mask, val_mask, test_mask, raw_w, 2);
    auto r2_pos  = train_and_evaluate_regime<Posit16_Type>("Regime 2", "Posit16", xr, er, edges, labels, train_mask, val_mask, test_mask, raw_w, 2);
    record("Regime 2 (Pure 16-Bit Adam)", "FP16", r2_fp16);
    record("Regime 2 (Pure 16-Bit Adam)", "Posit16", r2_pos);

    // 3. Regime 3: Double Adam
    cout << "\n[Regime 3: Double Optimizer (64-bit Adam States, 200 Epochs)]\n";
    auto r3_fp16 = train_and_evaluate_regime<FP16_Type>("Regime 3", "FP16", xr, er, edges, labels, train_mask, val_mask, test_mask, raw_w, 3);
    auto r3_pos  = train_and_evaluate_regime<Posit16_Type>("Regime 3", "Posit16", xr, er, edges, labels, train_mask, val_mask, test_mask, raw_w, 3);
    record("Regime 3 (Double Adam States)", "FP16", r3_fp16);
    record("Regime 3 (Double Adam States)", "Posit16", r3_pos);

    cout << "\n=========================================================================\n";
    cout << "  AUDIT COMPLETE! All results written to results/logs/numerical_accuracy_3regimes.csv\n";
    cout << "=========================================================================\n";

    return 0;
}

