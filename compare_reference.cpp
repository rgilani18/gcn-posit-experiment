#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <limits>
#include <string>

constexpr size_t COUNT = 2708 * 7;

template <typename T>
std::vector<T> load_binary(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open " + filename);
    }

    std::vector<T> values(COUNT);
    file.read(reinterpret_cast<char*>(values.data()),
              COUNT * sizeof(T));

    if (file.gcount() !=
        static_cast<std::streamsize>(COUNT * sizeof(T))) {
        throw std::runtime_error("Unexpected file size: " + filename);
    }

    return values;
}

void compare(const std::string& name,
             const std::vector<float>& output,
             const std::vector<long double>& reference) {
    long double max_error = 0.0L;
    long double sum_error = 0.0L;
    long double sum_squared_error = 0.0L;

    for (size_t i = 0; i < COUNT; ++i) {
        long double error =
            std::fabs(static_cast<long double>(output[i]) - reference[i]);

        max_error = std::max(max_error, error);
        sum_error += error;
        sum_squared_error += error * error;
    }

    std::cout << name << '\n'
              << "  Max absolute error: " << max_error << '\n'
              << "  Mean absolute error: " << sum_error / COUNT << '\n'
              << "  RMSE: "
              << std::sqrt(sum_squared_error / COUNT) << "\n\n";
}

int main() {
    try {
        auto reference =
            load_binary<long double>("golden_logits_longdouble.bin");
        auto fp32 = load_binary<float>("logits_fp32.bin");
        auto posit32 = load_binary<float>("logits_posit32.bin");

        for (size_t i = 0; i < COUNT; ++i) {
            if (!std::isfinite(reference[i]) ||
                !std::isfinite(fp32[i]) ||
                !std::isfinite(posit32[i])) {
                throw std::runtime_error("Non-finite value detected");
            }
        }

        std::cout << std::scientific << std::setprecision(10);
        std::cout << "Reference: long double\n"
                  << "Number of logits: " << COUNT << "\n\n";

        compare("Universal FP32", fp32, reference);
        compare("Posit32", posit32, reference);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
