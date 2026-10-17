#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>

using namespace std;

// --------------------------------------------------
// Constants
// --------------------------------------------------

const int NUM_NODES = 2708;
const int NUM_FEATURES = 1433;
const int HIDDEN = 16;
const int NUM_CLASSES = 7;
const int NUM_EDGES = 13264;

// --------------------------------------------------
// Binary file loader
// --------------------------------------------------

template <typename T>
vector<T> load_binary(const string& filename, size_t count)
{
    vector<T> data(count);

    ifstream file(filename, ios::binary);

    if (!file)
    {
        cerr << "Error opening: " << filename << endl;
        exit(1);
    }

    file.read(
        reinterpret_cast<char*>(data.data()),
        count * sizeof(T)
    );

    if (!file)
    {
        cerr << "Error reading: " << filename << endl;
        exit(1);
    }

    return data;
}

// --------------------------------------------------
// Argmax
// --------------------------------------------------

int argmax(const long double* values, int size)
{
    int index = 0;

    for (int i = 1; i < size; i++)
    {
        if (values[i] > values[index])
            index = i;
    }

    return index;
}

// --------------------------------------------------
// Main
// --------------------------------------------------

int main()
{
    cout << "Loading data..." << endl;

    // Input features
    auto features = load_binary<float>(
        "features_fp32.bin",
        NUM_NODES * NUM_FEATURES
    );

    // Normalized graph
    auto edge_index = load_binary<long long>(
        "norm_edge_index_int64.bin",
        2 * NUM_EDGES
    );

    auto edge_weight = load_binary<float>(
        "norm_edge_weight_fp32.bin",
        NUM_EDGES
    );

    // Model parameters
    auto w1 = load_binary<float>(
        "conv1_weight_fp32.bin",
        HIDDEN * NUM_FEATURES
    );

    auto b1 = load_binary<float>(
        "conv1_bias_fp32.bin",
        HIDDEN
    );

    auto w2 = load_binary<float>(
        "conv2_weight_fp32.bin",
        NUM_CLASSES * HIDDEN
    );

    auto b2 = load_binary<float>(
        "conv2_bias_fp32.bin",
        NUM_CLASSES
    );

    // Labels and test mask
    auto labels = load_binary<long long>(
        "labels_int64.bin",
        NUM_NODES
    );

    auto test_mask = load_binary<unsigned char>(
        "test_mask_uint8.bin",
        NUM_NODES
    );

    cout << "Data loaded successfully." << endl;

    // --------------------------------------------------
    // Layer 1: X @ W1^T
    // --------------------------------------------------

    vector<long double> support1(NUM_NODES * HIDDEN, 0.0L);

    for (int n = 0; n < NUM_NODES; n++)
    {
        for (int h = 0; h < HIDDEN; h++)
        {
            long double sum = 0.0L;

            for (int f = 0; f < NUM_FEATURES; f++)
            {
                sum +=
                    static_cast<long double>(features[n * NUM_FEATURES + f]) *
                    static_cast<long double>(w1[h * NUM_FEATURES + f]);
            }

            support1[n * HIDDEN + h] = sum;
        }
    }

    // --------------------------------------------------
    // Graph aggregation
    // --------------------------------------------------

    vector<long double> hidden(NUM_NODES * HIDDEN, 0.0L);

    for (int e = 0; e < NUM_EDGES; e++)
    {
        int src = static_cast<int>(edge_index[e]);
        int dst = static_cast<int>(edge_index[NUM_EDGES + e]);

        long double ew = edge_weight[e];

        for (int h = 0; h < HIDDEN; h++)
        {
            hidden[dst * HIDDEN + h] +=
                ew * support1[src * HIDDEN + h];
        }
    }

    // Bias + ReLU

    for (int n = 0; n < NUM_NODES; n++)
    {
        for (int h = 0; h < HIDDEN; h++)
        {
            long double value =
                hidden[n * HIDDEN + h] + (long double)b1[h];

            hidden[n * HIDDEN + h] =
                max((long double)0.0, value);
        }
    }

    // --------------------------------------------------
    // Layer 2: H @ W2^T
    // --------------------------------------------------

    vector<long double> support2(NUM_NODES * NUM_CLASSES, 0.0L);

    for (int n = 0; n < NUM_NODES; n++)
    {
        for (int c = 0; c < NUM_CLASSES; c++)
        {
            long double sum = 0.0L;

            for (int h = 0; h < HIDDEN; h++)
            {
                sum +=
                    hidden[n * HIDDEN + h] *
                    static_cast<long double>(w2[c * HIDDEN + h]);
            }

            support2[n * NUM_CLASSES + c] = sum;
        }
    }

    // --------------------------------------------------
    // Second graph aggregation
    // --------------------------------------------------

    vector<long double> logits(
        NUM_NODES * NUM_CLASSES,
        0.0L
    );

    for (int e = 0; e < NUM_EDGES; e++)
    {
        int src = static_cast<int>(edge_index[e]);
        int dst = static_cast<int>(edge_index[NUM_EDGES + e]);

        long double ew = edge_weight[e];

        for (int c = 0; c < NUM_CLASSES; c++)
        {
            logits[dst * NUM_CLASSES + c] +=
                ew * support2[src * NUM_CLASSES + c];
        }
    }

    // Add final bias

    for (int n = 0; n < NUM_NODES; n++)
    {
        for (int c = 0; c < NUM_CLASSES; c++)
        {
            logits[n * NUM_CLASSES + c] += b2[c];
        }
    }

    // --------------------------------------------------
    // Test accuracy
    // --------------------------------------------------

    int correct = 0;
    int total = 0;

    for (int n = 0; n < NUM_NODES; n++)
    {
        if (test_mask[n])
        {
            int prediction =
                argmax(
                    &logits[n * NUM_CLASSES],
                    NUM_CLASSES
                );

            if (prediction == labels[n])
                correct++;

            total++;
        }
    }

    double accuracy =
        static_cast<double>(correct) / total;

    cout << fixed << setprecision(6);

    cout << "\nC++ Long Double Reference Test Accuracy: "
         << accuracy << endl;

    cout << "Correct: "
         << correct << " / "
         << total << endl;

    // --------------------------------------------------
    // Compare with PyTorch golden logits
    // --------------------------------------------------

    auto golden = load_binary<float>(
        "golden_logits_fp32.bin",
        NUM_NODES * NUM_CLASSES
    );

    float max_abs_error = 0.0f;
    double sum_abs_error = 0.0;

    for (size_t i = 0; i < logits.size(); i++)
    {
        float error =
            fabs(logits[i] - golden[i]);

        max_abs_error =
            max(max_abs_error, error);

        sum_abs_error += error;
    }

    double mean_abs_error =
        sum_abs_error / logits.size();

    cout << "\nComparison with PyTorch:" << endl;

    cout << "Maximum absolute error: "
         << max_abs_error << endl;

    cout << "Mean absolute error: "
         << mean_abs_error << endl;
        // Save long-double reference logits
    ofstream reference_file(
        "golden_logits_longdouble.bin",
        ios::binary
    );

    reference_file.write(
        reinterpret_cast<const char*>(logits.data()),
        logits.size() * sizeof(long double)
    );

    reference_file.close();

    cout << "Saved: golden_logits_longdouble.bin" << endl;
    return 0;
}
