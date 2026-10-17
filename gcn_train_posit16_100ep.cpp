#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <sw/universal/number/cfloat/cfloat.hpp>
#include <sw/universal/number/posit/posit.hpp>

using namespace std;
using Posit16 = sw::universal::posit<16, 1>;

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

    if (file.gcount() != static_cast<streamsize>(count * sizeof(T))) {
        cerr << "Incorrect file size: " << filename << '\n';
        exit(1);
    }

    return data;
}

// Save weights as float32 values in this order:
// W1, b1, W2, b2.
bool save_weights(const string& filename,
                  const vector<Posit16>& w1,
                  const vector<Posit16>& b1,
                  const vector<Posit16>& w2,
                  const vector<Posit16>& b2) {
    ofstream file(filename, ios::binary);

    if (!file) {
        cerr << "Cannot write " << filename << '\n';
        return false;
    }

    auto write_vector = [&](const vector<Posit16>& values) {
        for (const auto& value : values) {
            float v = static_cast<float>(static_cast<double>(value));
            file.write(reinterpret_cast<const char*>(&v), sizeof(v));
        }
    };

    write_vector(w1);
    write_vector(b1);
    write_vector(w2);
    write_vector(b2);

    return static_cast<bool>(file);
}

int main(int argc, char* argv[]) {
    int epochs = 100;
    const Posit16 learning_rate = Posit16(0.01);

    if (argc > 1) {
        try {
            epochs = stoi(argv[1]);
        } catch (...) {
            cerr << "Usage: " << argv[0] << " [epochs]\n";
            return 1;
        }
    }

    if (epochs < 1) {
        cerr << "Epoch count must be at least 1\n";
        return 1;
    }

    // Load the same Cora inputs used by the gradient-check program.
    auto xr = load_binary<float>("features_fp32.bin", N * F);
    auto er = load_binary<float>("norm_edge_weight_fp32.bin", E);
    auto edges = load_binary<int64_t>("norm_edge_index_int64.bin", 2 * E);
    auto labels = load_binary<int64_t>("labels_int64.bin", N);
    auto mask = load_binary<uint8_t>("train_mask_uint8.bin", N);
    auto val_mask = load_binary<uint8_t>("val_mask_uint8.bin", N);

    vector<Posit16> x(N * F), ew(E);
    vector<Posit16> w1(H * F), b1(H, Posit16(0));
    vector<Posit16> w2(C * H), b2(C, Posit16(0));

    // Adam first and second moments for all parameters.
    vector<Posit16> mw1(w1.size(), Posit16(0)), vw1(w1.size(), Posit16(0));
    vector<Posit16> mb1(b1.size(), Posit16(0)), vb1(b1.size(), Posit16(0));
    vector<Posit16> mw2(w2.size(), Posit16(0)), vw2(w2.size(), Posit16(0));
    vector<Posit16> mb2(b2.size(), Posit16(0)), vb2(b2.size(), Posit16(0));

    const Posit16 beta1 = Posit16(0.9);
    const Posit16 beta2 = Posit16(0.999);
    const Posit16 adam_eps = Posit16(1e-4);
    const Posit16 weight_decay = Posit16(5e-4);

    for (size_t i = 0; i < x.size(); ++i)
        x[i] = Posit16(xr[i]);

    for (int e = 0; e < E; ++e)
        ew[e] = Posit16(er[e]);

    // Validate graph indices and labels before training.
    for (int e = 0; e < E; ++e) {
        int64_t src = edges[e];
        int64_t dst = edges[E + e];

        if (src < 0 || src >= N || dst < 0 || dst >= N) {
            cerr << "Invalid edge index at edge " << e << '\n';
            return 1;
        }
    }

    int train_count = 0;
    for (int n = 0; n < N; ++n) {
        if (mask[n]) {
            if (labels[n] < 0 || labels[n] >= C) {
                cerr << "Invalid label at node " << n << '\n';
                return 1;
            }
            ++train_count;
        }
    }

    if (train_count == 0) {
        cerr << "No training nodes found\n";
        return 1;
    }

    int val_count = 0;
    for (int n = 0; n < N; ++n) {
        if (val_mask[n]) {
            if (labels[n] < 0 || labels[n] >= C) {
                cerr << "Invalid validation label at node " << n << '\n';
                return 1;
            }
            ++val_count;
        }
    }

    if (val_count == 0) {
        cerr << "No validation nodes found\n";
        return 1;
    }

    // Reproducible initialization, matching the gradient-check program.
    mt19937 rng(42);
    normal_distribution<float> dist(0.0f, 1.0f);

    for (auto& w : w1)
        w = Posit16(dist(rng) * sqrt(2.0f / F));

    for (auto& w : w2)
        w = Posit16(dist(rng) * sqrt(2.0f / H));

    if (!save_weights("gcn_initial_weights_posit16_100ep.bin", w1, b1, w2, b2))
        return 1;

    cout << "Universal Posit16 GCN training\n";
    cout << "Nodes: " << N << ", features: " << F
         << ", hidden units: " << H << ", classes: " << C << '\n';
    cout << "Edges: " << E << ", training nodes: " << train_count << '\n';
    cout << "Epochs: " << epochs
         << ", learning rate: " << learning_rate << '\n';
    cout << "Initial weights saved to gcn_initial_weights_posit16_100ep.bin\n";

    ofstream csv_log("training_log_posit16_100ep.csv");
    if (!csv_log) {
        cerr << "Cannot open training_log_posit16_100ep.csv\n";
        return 1;
    }
    csv_log << "epoch,train_loss_before_update,train_accuracy_before_update,"
               "val_loss_after_update,val_accuracy_after_update\n";

    const int patience = 15;
    int epochs_without_improvement = 0;
    int best_epoch = 0;
    double best_val_loss = INFINITY;

    vector<Posit16> best_w1, best_b1, best_w2, best_b2;

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        // ------------------------------------------------------------
        // 1. Forward pass: first graph-convolution layer.
        // support1 = X * W1^T
        // hidden = ReLU(A * support1 + b1)
        // ------------------------------------------------------------
        vector<Posit16> support1(N * H, Posit16(0));

        for (int n = 0; n < N; ++n) {
            for (int h = 0; h < H; ++h) {
                Posit16 sum = Posit16(0);

                for (int f = 0; f < F; ++f)
                    sum += x[n * F + f] * w1[h * F + f];

                support1[n * H + h] = sum;
            }
        }

        vector<Posit16> aggregated1(N * H, Posit16(0));

        for (int e = 0; e < E; ++e) {
            int src = static_cast<int>(edges[e]);
            int dst = static_cast<int>(edges[E + e]);

            for (int h = 0; h < H; ++h)
                aggregated1[dst * H + h] +=
                    ew[e] * support1[src * H + h];
        }

        vector<Posit16> pre_relu(N * H, Posit16(0));
        vector<Posit16> hidden(N * H, Posit16(0));

        for (int n = 0; n < N; ++n) {
            for (int h = 0; h < H; ++h) {
                Posit16 v = aggregated1[n * H + h] + b1[h];
                pre_relu[n * H + h] = v;
                hidden[n * H + h] = v > Posit16(0) ? v : Posit16(0);
            }
        }

        // ------------------------------------------------------------
        // 2. Forward pass: second graph-convolution layer.
        // support2 = hidden * W2^T
        // logits = A * support2 + b2
        // ------------------------------------------------------------
        vector<Posit16> support2(N * C, Posit16(0));

        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                Posit16 sum = Posit16(0);

                for (int h = 0; h < H; ++h)
                    sum += hidden[n * H + h] * w2[c * H + h];

                support2[n * C + c] = sum;
            }
        }

        vector<Posit16> logits(N * C, Posit16(0));

        for (int e = 0; e < E; ++e) {
            int src = static_cast<int>(edges[e]);
            int dst = static_cast<int>(edges[E + e]);

            for (int c = 0; c < C; ++c)
                logits[dst * C + c] +=
                    ew[e] * support2[src * C + c];
        }

        for (int n = 0; n < N; ++n)
            for (int c = 0; c < C; ++c)
                logits[n * C + c] += b2[c];

        // ------------------------------------------------------------
        // 3. Stable mean cross-entropy and output gradients.
        // Softmax/loss and gradient accumulators use double.
        // ------------------------------------------------------------
        Posit16 loss = Posit16(0);
        vector<Posit16> dlogits(N * C, Posit16(0));
        int correct = 0;

        for (int n = 0; n < N; ++n) {
            if (!mask[n]) continue;

            int y = static_cast<int>(labels[n]);
            int predicted = 0;

            for (int c = 1; c < C; ++c) {
                if (static_cast<double>(logits[n * C + c]) >
                    static_cast<double>(logits[n * C + predicted])) {
                    predicted = c;
                }
            }

            if (predicted == y)
                ++correct;

            double max_logit = static_cast<double>(logits[n * C]);

            for (int c = 1; c < C; ++c)
                max_logit = max(
                    max_logit,
                    static_cast<double>(logits[n * C + c])
                );

            double exp_sum = 0.0;

            for (int c = 0; c < C; ++c)
                exp_sum += exp(
                    static_cast<double>(logits[n * C + c]) - max_logit
                );

            loss += max_logit + log(exp_sum)
                  - static_cast<double>(logits[n * C + y]);

            for (int c = 0; c < C; ++c) {
                double probability =
                    exp(static_cast<double>(logits[n * C + c]) - max_logit)
                    / exp_sum;

                dlogits[n * C + c] =
                    probability - (c == y ? 1.0 : 0.0);
            }
        }

        loss /= train_count;

        for (int n = 0; n < N; ++n) {
            if (!mask[n]) continue;

            for (int c = 0; c < C; ++c)
                dlogits[n * C + c] /= train_count;
        }

        double train_accuracy =
            static_cast<double>(correct) / train_count;

        // ------------------------------------------------------------
        // 4. Backpropagate through the second graph aggregation.
        // ------------------------------------------------------------
        vector<Posit16> dSupport2(N * C, Posit16(0));

        for (int e = 0; e < E; ++e) {
            int src = static_cast<int>(edges[e]);
            int dst = static_cast<int>(edges[E + e]);

            for (int c = 0; c < C; ++c)
                dSupport2[src * C + c] +=
                    ew[e] * dlogits[dst * C + c];
        }

        vector<Posit16> dW2(C * H, Posit16(0));
        vector<Posit16> db2(C, Posit16(0));

        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                db2[c] += dlogits[n * C + c];

                for (int h = 0; h < H; ++h)
                    dW2[c * H + h] +=
                        dSupport2[n * C + c] * hidden[n * H + h];
            }
        }

        // Backpropagate through W2 and the ReLU activation.
        vector<Posit16> dHidden(N * H, Posit16(0));

        for (int n = 0; n < N; ++n) {
            for (int h = 0; h < H; ++h) {
                Posit16 g = Posit16(0);

                for (int c = 0; c < C; ++c)
                    g += dSupport2[n * C + c] * w2[c * H + h];

                dHidden[n * H + h] =
                    pre_relu[n * H + h] > Posit16(0) ? g : 0.0;
            }
        }

        // Backpropagate through the first graph aggregation.
        vector<Posit16> dSupport1(N * H, Posit16(0));

        for (int e = 0; e < E; ++e) {
            int src = static_cast<int>(edges[e]);
            int dst = static_cast<int>(edges[E + e]);

            for (int h = 0; h < H; ++h)
                dSupport1[src * H + h] +=
                    ew[e] * dHidden[dst * H + h];
        }

        vector<Posit16> dW1(H * F, Posit16(0));
        vector<Posit16> db1(H, Posit16(0));

        for (int n = 0; n < N; ++n) {
            for (int h = 0; h < H; ++h) {
                db1[h] += dHidden[n * H + h];

                for (int f = 0; f < F; ++f)
                    dW1[h * F + f] +=
                        dSupport1[n * H + h] * x[n * F + f];
            }
        }

        // Reject invalid gradients instead of silently updating weights.
        bool gradients_finite = true;

        for (const Posit16& g : dW1)
            gradients_finite = gradients_finite && isfinite(g);
        for (const Posit16& g : db1)
            gradients_finite = gradients_finite && isfinite(g);
        for (const Posit16& g : dW2)
            gradients_finite = gradients_finite && isfinite(g);
        for (const Posit16& g : db2)
            gradients_finite = gradients_finite && isfinite(g);

        if (!gradients_finite || !isfinite(loss)) {
            cerr << "Non-finite loss or gradient at epoch "
                 << epoch << '\n';
            return 1;
        }

        // ------------------------------------------------------------
        // 5. Adam update with coupled L2 weight decay.
        // Matches PyTorch Adam defaults: betas=(0.9, 0.999), eps=1e-4.
        // Weight decay applies to weights and biases, as in model.parameters().
        // ------------------------------------------------------------
        const Posit16 bias_correction1 =
            Posit16(1.0 - std::pow(0.9, epoch));
        const Posit16 bias_correction2 =
            Posit16(1.0 - std::pow(0.999, epoch));

        auto adam_update = [&](vector<Posit16>& param,
                               const vector<Posit16>& grad,
                               vector<Posit16>& m,
                               vector<Posit16>& v) {
            for (size_t i = 0; i < param.size(); ++i) {
                Posit16 g = grad[i] + weight_decay * param[i];

                m[i] = beta1 * m[i] + (Posit16(1) - beta1) * g;
                v[i] = beta2 * v[i] +
                       (Posit16(1) - beta2) * g * g;

                Posit16 m_hat = m[i] / bias_correction1;
                Posit16 v_hat = v[i] / bias_correction2;
                Posit16 denominator =
                    Posit16(std::sqrt(static_cast<double>(v_hat))) + adam_eps;

                param[i] = param[i] -
                           learning_rate * m_hat / denominator;
            }
        };

        adam_update(w1, dW1, mw1, vw1);
        adam_update(b1, db1, mb1, vb1);
        adam_update(w2, dW2, mw2, vw2);
        adam_update(b2, db2, mb2, vb2);

        // Validation forward pass using the updated weights.
        vector<Posit16> vs1(N * H, Posit16(0));
        for (int n = 0; n < N; ++n)
            for (int h = 0; h < H; ++h)
                for (int f = 0; f < F; ++f)
                    vs1[n * H + h] += x[n * F + f] * w1[h * F + f];

        vector<Posit16> va1(N * H, Posit16(0));
        for (int e = 0; e < E; ++e) {
            int src = static_cast<int>(edges[e]);
            int dst = static_cast<int>(edges[E + e]);
            for (int h = 0; h < H; ++h)
                va1[dst * H + h] += ew[e] * vs1[src * H + h];
        }

        vector<Posit16> vh(N * H, Posit16(0));
        for (int n = 0; n < N; ++n)
            for (int h = 0; h < H; ++h) {
                Posit16 z = va1[n * H + h] + b1[h];
                vh[n * H + h] = z > Posit16(0) ? z : Posit16(0);
            }

        vector<Posit16> vs2(N * C, Posit16(0));
        for (int n = 0; n < N; ++n)
            for (int c = 0; c < C; ++c)
                for (int h = 0; h < H; ++h)
                    vs2[n * C + c] += vh[n * H + h] * w2[c * H + h];

        vector<Posit16> vl(N * C, Posit16(0));
        for (int e = 0; e < E; ++e) {
            int src = static_cast<int>(edges[e]);
            int dst = static_cast<int>(edges[E + e]);
            for (int c = 0; c < C; ++c)
                vl[dst * C + c] += ew[e] * vs2[src * C + c];
        }
        for (int n = 0; n < N; ++n)
            for (int c = 0; c < C; ++c)
                vl[n * C + c] += b2[c];

        double val_loss = 0.0;
        int val_correct = 0;
        for (int n = 0; n < N; ++n) {
            if (!val_mask[n]) continue;

            int y = static_cast<int>(labels[n]);
            int pred = 0;
            double max_logit = static_cast<double>(vl[n * C]);
            for (int c = 1; c < C; ++c) {
                double z = static_cast<double>(vl[n * C + c]);
                if (z > static_cast<double>(vl[n * C + pred])) pred = c;
                max_logit = max(max_logit, z);
            }
            if (pred == y) ++val_correct;

            double exp_sum = 0.0;
            for (int c = 0; c < C; ++c)
                exp_sum += exp(static_cast<double>(vl[n * C + c]) - max_logit);
            val_loss += max_logit + log(exp_sum)
                      - static_cast<double>(vl[n * C + y]);
        }
        val_loss /= val_count;
        double val_accuracy = static_cast<double>(val_correct) / val_count;

        csv_log << epoch << "," << loss << "," << train_accuracy << ","
                << val_loss << "," << val_accuracy << "\n";
        if (!csv_log) {
            cerr << "Error writing training_log_posit16_100ep.csv\n";
            return 1;
        }

        if (isfinite(val_loss) && val_loss < best_val_loss) {
            best_val_loss = val_loss;
            best_epoch = epoch;
            epochs_without_improvement = 0;
            best_w1 = w1;
            best_b1 = b1;
            best_w2 = w2;
            best_b2 = b2;
        } else {
            ++epochs_without_improvement;
        }

        if (epoch == 1 || epoch % 10 == 0 || epoch == epochs ||
            epochs_without_improvement >= patience) {
            cout << "Epoch " << epoch << "/" << epochs
                 << " | train loss before update: " << loss
                 << " | train accuracy before update: " << train_accuracy
                 << " | validation loss: " << val_loss
                 << " | validation accuracy: " << val_accuracy
                 << " | best epoch: " << best_epoch << '\n';
        }

        if (epochs_without_improvement >= patience) {
            cout << "Early stopping: validation loss did not improve for "
                 << patience << " epochs.\n";
            break;
        }
    }

    if (best_epoch == 0) {
        cerr << "No valid best checkpoint selected.\n";
        return 1;
    }

    w1 = best_w1;
    b1 = best_b1;
    w2 = best_w2;
    b2 = best_b2;

    if (!save_weights("gcn_trained_weights_posit16_100ep.bin", w1, b1, w2, b2))
        return 1;

    cout << "Training complete. Best validation epoch: " << best_epoch << '\n';
    cout << "Best validation loss: " << best_val_loss << '\n';
    cout << "Final weights saved to gcn_trained_weights_posit16_100ep.bin\n";
    cout << "Initial weights saved to gcn_initial_weights_posit16_100ep.bin\n";

    return 0;
}
