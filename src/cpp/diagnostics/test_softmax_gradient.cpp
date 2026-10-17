#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

int main() {
    constexpr int C = 7;
    constexpr int train_nodes = 2;

    // Two example nodes, each with seven class logits.
    const double logits[train_nodes][C] = {
        {1.0, 2.0, 0.5, -1.0, 0.0, 0.3, -0.2},
        {0.2, -0.1, 1.5, 0.0, 0.7, -0.5, 0.1}
    };
    const int labels[train_nodes] = {1, 2};

    double dlogits[train_nodes][C] = {};
    double loss = 0.0;

    for (int n = 0; n < train_nodes; ++n) {
        double max_logit = logits[n][0];
        for (int c = 1; c < C; ++c)
            max_logit = std::max(max_logit, logits[n][c]);

        double exp_sum = 0.0;
        for (int c = 0; c < C; ++c)
            exp_sum += std::exp(logits[n][c] - max_logit);

        const double log_sum_exp = max_logit + std::log(exp_sum);
        loss += log_sum_exp - logits[n][labels[n]];

        for (int c = 0; c < C; ++c) {
            const double probability =
                std::exp(logits[n][c] - max_logit) / exp_sum;
            dlogits[n][c] =
                (probability - (c == labels[n] ? 1.0 : 0.0))
                / train_nodes;
        }
    }

    loss /= train_nodes;

    bool finite = std::isfinite(loss);
    for (int n = 0; n < train_nodes; ++n) {
        double row_sum = 0.0;
        for (int c = 0; c < C; ++c) {
            finite = finite && std::isfinite(dlogits[n][c]);
            row_sum += dlogits[n][c];
        }
        std::cout << "Node " << n
                  << " gradient sum: " << row_sum << '\n';
    }

    std::cout << "Mean cross-entropy: " << loss << '\n';
    std::cout << "All values finite: " << (finite ? "YES" : "NO") << '\n';

    if (!finite) return 1;
    return 0;
}
