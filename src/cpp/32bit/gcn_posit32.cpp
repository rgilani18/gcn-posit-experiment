#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <universal/number/posit/posit.hpp>
using namespace std;
using Posit32 = sw::universal::posit<32, 2>;
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

int argmax(const Posit32* values, int size)
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

    vector<Posit32> support1(NUM_NODES * HIDDEN, Posit32(0));

    for (int n = 0; n < NUM_NODES; n++)
    {
        for (int h = 0; h < HIDDEN; h++)
        {
           Posit32 sum = Posit32(0);
           for (int f = 0; f < NUM_FEATURES; f++)
            {
                sum +=
    Posit32(features[n * NUM_FEATURES + f]) *
    Posit32(w1[h * NUM_FEATURES + f]);
            }

            support1[n * HIDDEN + h] = sum;
        }
    }

    // --------------------------------------------------
    // Graph aggregation
    // --------------------------------------------------

    vector<Posit32> hidden(NUM_NODES * HIDDEN, Posit32(0));

    for (int e = 0; e < NUM_EDGES; e++)
    {
        int src = static_cast<int>(edge_index[e]);
        int dst = static_cast<int>(edge_index[NUM_EDGES + e]);

        Posit32 ew = Posit32(edge_weight[e]);

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
            Posit32 value =
    		hidden[n * HIDDEN + h] + Posit32(b1[h]);

            hidden[n * HIDDEN + h] =
                max(Posit32(0), value);
        }
    }

    // --------------------------------------------------
    // Layer 2: H @ W2^T
    // --------------------------------------------------

    vector<Posit32> support2(NUM_NODES * NUM_CLASSES, Posit32(0));

    for (int n = 0; n < NUM_NODES; n++)
    {
        for (int c = 0; c < NUM_CLASSES; c++)
        {
            Posit32 sum = Posit32(0);

            for (int h = 0; h < HIDDEN; h++)
            {
                sum +=
                    Posit32(hidden[n * HIDDEN + h]) *
                    Posit32(w2[c * HIDDEN + h]);
            }

            support2[n * NUM_CLASSES + c] = sum;
        }
    }

    // --------------------------------------------------
    // Second graph aggregation
    // --------------------------------------------------

    vector<Posit32> logits(
    NUM_NODES * NUM_CLASSES,
    Posit32(0)
	);

    for (int e = 0; e < NUM_EDGES; e++)
    {
        int src = static_cast<int>(edge_index[e]);
        int dst = static_cast<int>(edge_index[NUM_EDGES + e]);

        Posit32 ew = Posit32(edge_weight[e]);

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

    cout << "\nC++ Posit32 Test Accuracy: "
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
    fabs(
        static_cast<float>(logits[i]) - golden[i]
    );

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
        ofstream posit_file(
        "logits_posit32.bin",
        ios::binary
    );

    for (const auto& value : logits)
    {
        float output = static_cast<float>(value);

        posit_file.write(
            reinterpret_cast<const char*>(&output),
            sizeof(float)
        );
    }

    posit_file.close();

    cout << "Saved: logits_posit32.bin" << endl;
    return 0;
}
