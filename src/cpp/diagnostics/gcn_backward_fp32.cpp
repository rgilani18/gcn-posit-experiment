#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <random>
#include <string>

#include <sw/universal/number/cfloat/cfloat.hpp>

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
    vector<FP32> pre_relu(N * H, FP32(0));
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
            pre_relu[n * H + h] = v;
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

        // Mean cross-entropy and output gradient dL/dlogits.
    // Loss/softmax use double for numerical stability;
    // logits themselves are computed using Universal FP32.
    vector<double> dlogits(N * C, 0.0);
    double loss = 0.0;
    int count = 0;
    double max_abs_gradient = 0.0;
    double max_abs_row_sum = 0.0;

    for (int n = 0; n < N; ++n) {
        if (!mask[n]) continue;

        int y = static_cast<int>(labels[n]);
        if (y < 0 || y >= C) {
            cerr << "Invalid label\n";
            return 1;
        }

        double maxlogit = static_cast<double>(logits[n * C]);
        for (int c = 1; c < C; ++c)
            maxlogit = max(
                maxlogit,
                static_cast<double>(logits[n * C + c])
            );

        double exp_sum = 0.0;
        for (int c = 0; c < C; ++c)
            exp_sum += exp(
                static_cast<double>(logits[n * C + c]) - maxlogit
            );

        double log_sum_exp = maxlogit + log(exp_sum);
        loss += log_sum_exp - static_cast<double>(logits[n * C + y]);

        double row_sum = 0.0;
        for (int c = 0; c < C; ++c) {
            double p = exp(
                static_cast<double>(logits[n * C + c]) - maxlogit
            ) / exp_sum;

            double g = (p - (c == y ? 1.0 : 0.0)) ;
            dlogits[n * C + c] = g;
            row_sum += g;
        }

        max_abs_row_sum = max(max_abs_row_sum, abs(row_sum));
        ++count;
    }

    if (count == 0) {
        cerr << "No training nodes found\n";
        return 1;
    }

    loss /= count;

    // Mean-loss gradient: divide each training-node gradient by count.
    for (int n = 0; n < N; ++n) {
        if (!mask[n]) continue;
        for (int c = 0; c < C; ++c) {
            dlogits[n * C + c] /= count;
            max_abs_gradient = max(
                max_abs_gradient,
                abs(dlogits[n * C + c])
            );
        }
    }

    // Non-training nodes must have zero output gradient.
    bool nontrain_zero = true;
    for (int n = 0; n < N; ++n) {
        if (mask[n]) continue;
        for (int c = 0; c < C; ++c) {
            if (dlogits[n * C + c] != 0.0)
                nontrain_zero = false;
        }
    }


    // Backpropagate through the second graph aggregation.
    // Forward: logits[dst] += ew * support2[src]
    // Backward: dSupport2[src] += ew * dLogits[dst]
    vector<double> dSupport2(N * C, 0.0);
    for (int e = 0; e < E; ++e) {
        int src = static_cast<int>(edges[e]);
        int dst = static_cast<int>(edges[E + e]);
        for (int c = 0; c < C; ++c)
            dSupport2[src * C + c] +=
                static_cast<double>(ew[e]) * dlogits[dst * C + c];
    }

    vector<double> dW2(C * H, 0.0), db2(C, 0.0);
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            db2[c] += dlogits[n * C + c];
            for (int h = 0; h < H; ++h)
                dW2[c * H + h] +=
                    dSupport2[n * C + c] *
                    static_cast<double>(hidden[n * H + h]);
        }
    }

    double max_abs_dW2 = 0.0, max_abs_db2 = 0.0;
    
    // Numerical gradient check for W2[0].
    // Reuse hidden activations and recompute the second-layer forward pass.
    auto loss_for_w2_0 = [&](FP32 candidate_weight) -> double {
        FP32 original_weight = w2[0];
        w2[0] = candidate_weight;

        vector<FP32> candidate_support2(N * C, FP32(0));
        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                FP32 sum = FP32(0);
                for (int h = 0; h < H; ++h)
                    sum += hidden[n * H + h] * w2[c * H + h];
                candidate_support2[n * C + c] = sum;
            }
        }

        vector<FP32> candidate_logits(N * C, FP32(0));
        for (int e = 0; e < E; ++e) {
            int src = static_cast<int>(edges[e]);
            int dst = static_cast<int>(edges[E + e]);
            for (int c = 0; c < C; ++c)
                candidate_logits[dst * C + c] +=
                    ew[e] * candidate_support2[src * C + c];
        }

        for (int n = 0; n < N; ++n)
            for (int c = 0; c < C; ++c)
                candidate_logits[n * C + c] += b2[c];

        double candidate_loss = 0.0;
        for (int n = 0; n < N; ++n) {
            if (!mask[n]) continue;

            int y = static_cast<int>(labels[n]);
            double max_logit =
                static_cast<double>(candidate_logits[n * C]);

            for (int c = 1; c < C; ++c)
                max_logit = max(
                    max_logit,
                    static_cast<double>(candidate_logits[n * C + c])
                );

            double exp_sum = 0.0;
            for (int c = 0; c < C; ++c)
                exp_sum += exp(
                    static_cast<double>(candidate_logits[n * C + c])
                    - max_logit
                );

            candidate_loss += max_logit + log(exp_sum)
                - static_cast<double>(candidate_logits[n * C + y]);
        }

        w2[0] = original_weight;
        return candidate_loss / count;
    };

    const double epsilon = 1e-2;
    FP32 original_w2_0 = w2[0];
    double loss_plus = loss_for_w2_0(
        FP32(static_cast<double>(original_w2_0) + epsilon)
    );
    double loss_minus = loss_for_w2_0(
        FP32(static_cast<double>(original_w2_0) - epsilon)
    );

    double numerical_dW2_0 = (loss_plus - loss_minus) / (2.0 * epsilon);

    cout << "W2[0] finite-difference gradient check\n";
    cout << "Analytical gradient: " << dW2[0] << '\n';
    cout << "Numerical gradient:  " << numerical_dW2_0 << '\n';
    cout << "Absolute difference: "
         << abs(dW2[0] - numerical_dW2_0) << '\n';

    bool gradients_finite = true;
    for (double g : dW2) {
        gradients_finite = gradients_finite && isfinite(g);
        max_abs_dW2 = max(max_abs_dW2, abs(g));
    }
    for (double g : db2) {
        gradients_finite = gradients_finite && isfinite(g);
        max_abs_db2 = max(max_abs_db2, abs(g));
    }


    // Backpropagate from layer 2 to hidden activations.
    vector<double> dHidden(N * H, 0.0);
    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < H; ++h) {
            double g = 0.0;
            for (int c = 0; c < C; ++c)
                g += dSupport2[n * C + c] *
                     static_cast<double>(w2[c * H + h]);

            // ReLU derivative uses the pre-activation value.
            dHidden[n * H + h] =
                pre_relu[n * H + h] > FP32(0) ? g : 0.0;
        }
    }

    // Backpropagate through the first graph aggregation.
    vector<double> dSupport1(N * H, 0.0);
    for (int e = 0; e < E; ++e) {
        int src = static_cast<int>(edges[e]);
        int dst = static_cast<int>(edges[E + e]);
        for (int h = 0; h < H; ++h)
            dSupport1[src * H + h] +=
                static_cast<double>(ew[e]) * dHidden[dst * H + h];
    }

    vector<double> dW1(H * F, 0.0), db1(H, 0.0);
    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < H; ++h) {
            db1[h] += dHidden[n * H + h];
            for (int f = 0; f < F; ++f)
                dW1[h * F + f] +=
                    dSupport1[n * H + h] *
                    static_cast<double>(x[n * F + f]);
        }
    }
	
    // Numerical gradient check for W1[0].
    auto loss_for_w1_0 = [&](FP32 candidate_weight) -> double {
        FP32 original_weight = w1[0];
        w1[0] = candidate_weight;

        // Recompute first-layer support.
        vector<FP32> s1(N * H, FP32(0));
        for (int n = 0; n < N; ++n) {
            for (int h = 0; h < H; ++h) {
                FP32 sum = FP32(0);
                for (int f = 0; f < F; ++f)
                    sum += x[n * F + f] * w1[h * F + f];
                s1[n * H + h] = sum;
            }
        }

        // Graph aggregation, bias, and ReLU.
        vector<FP32> h1(N * H, FP32(0));
        for (int e = 0; e < E; ++e) {
            int src = static_cast<int>(edges[e]);
            int dst = static_cast<int>(edges[E + e]);
            for (int h = 0; h < H; ++h)
                h1[dst * H + h] += ew[e] * s1[src * H + h];
        }
        for (int n = 0; n < N; ++n)
            for (int h = 0; h < H; ++h) {
                FP32 v = h1[n * H + h] + b1[h];
                h1[n * H + h] = v > FP32(0) ? v : FP32(0);
            }

        // Second-layer support.
        vector<FP32> s2(N * C, FP32(0));
        for (int n = 0; n < N; ++n)
            for (int c = 0; c < C; ++c) {
                FP32 sum = FP32(0);
                for (int h = 0; h < H; ++h)
                    sum += h1[n * H + h] * w2[c * H + h];
                s2[n * C + c] = sum;
            }

        // Final graph aggregation and logits.
        vector<FP32> z(N * C, FP32(0));
        for (int e = 0; e < E; ++e) {
            int src = static_cast<int>(edges[e]);
            int dst = static_cast<int>(edges[E + e]);
            for (int c = 0; c < C; ++c)
                z[dst * C + c] += ew[e] * s2[src * C + c];
        }
        for (int n = 0; n < N; ++n)
            for (int c = 0; c < C; ++c)
                z[n * C + c] += b2[c];

        // Mean cross-entropy over training nodes.
        double total_loss = 0.0;
        for (int n = 0; n < N; ++n) {
            if (!mask[n]) continue;

            int y = static_cast<int>(labels[n]);
            double mx = static_cast<double>(z[n * C]);
            for (int c = 1; c < C; ++c)
                mx = max(mx, static_cast<double>(z[n * C + c]));

            double esum = 0.0;
            for (int c = 0; c < C; ++c)
                esum += exp(static_cast<double>(z[n * C + c]) - mx);

            total_loss += mx + log(esum)
                - static_cast<double>(z[n * C + y]);
        }

        w1[0] = original_weight;
        return total_loss / count;
    };

    const double epsilon_w1 = 1e-2;
    FP32 original_w1_0 = w1[0];
    double loss_w1_plus = loss_for_w1_0(
        FP32(static_cast<double>(original_w1_0) + epsilon_w1)
    );
    double loss_w1_minus = loss_for_w1_0(
        FP32(static_cast<double>(original_w1_0) - epsilon_w1)
    );
    double numerical_dW1_0 =
        (loss_w1_plus - loss_w1_minus) / (2.0 * epsilon_w1);

    cout << "W1[0] finite-difference gradient check\n";
    cout << "Analytical gradient: " << dW1[0] << '\n';
    cout << "Numerical gradient:  " << numerical_dW1_0 << '\n';
    cout << "Absolute difference: "
         << abs(dW1[0] - numerical_dW1_0) << '\n';

    double max_abs_dW1 = 0.0, max_abs_db1 = 0.0;
    for (double g : dW1) {
        gradients_finite = gradients_finite && isfinite(g);
        max_abs_dW1 = max(max_abs_dW1, abs(g));
    }
    for (double g : db1) {
        gradients_finite = gradients_finite && isfinite(g);
        max_abs_db1 = max(max_abs_db1, abs(g));
    }

    cout << "First-layer gradient check\n";
    cout << "dW1 elements: " << dW1.size() << '\n';
    cout << "db1 elements: " << db1.size() << '\n';
    cout << "Max |dW1|: " << max_abs_dW1 << '\n';
    cout << "Max |db1|: " << max_abs_db1 << '\n';
    cout << "All layer gradients finite: "
         << (gradients_finite ? "YES" : "NO") << '\n';

    cout << "Second-layer gradient check\n";
    cout << "dW2 elements: " << dW2.size() << '\n';
    cout << "db2 elements: " << db2.size() << '\n';
    cout << "Max |dW2|: " << max_abs_dW2 << '\n';
    cout << "Max |db2|: " << max_abs_db2 << '\n';
    cout << "Second-layer gradients finite: "
         << (gradients_finite ? "YES" : "NO") << '\n';

    cout << "Universal FP32 GCN output-gradient check\n";
    cout << "Training nodes: " << count << '\n';
    cout << "Mean cross-entropy: " << loss << '\n';
    cout << "Max absolute output gradient: " << max_abs_gradient << '\n';
    cout << "Max absolute pre-mean gradient row sum: "
         << max_abs_row_sum << '\n';
    cout << "Non-training gradients zero: "
         << (nontrain_zero ? "YES" : "NO") << '\n';
    cout << "No weight updates performed.\n";

    if (!nontrain_zero || !isfinite(loss) ||
        !isfinite(max_abs_gradient) || !isfinite(max_abs_row_sum))
        return 1;

    return 0;
}
