#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <limits>
#include <iomanip>

int main() {
    constexpr size_t COUNT = 2708 * 7;

    std::ifstream file("golden_logits_longdouble.bin", std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open reference file\n";
        return 1;
    }

    std::vector<long double> values(COUNT);
    file.read(reinterpret_cast<char*>(values.data()),
              COUNT * sizeof(long double));

    if (file.gcount() != static_cast<std::streamsize>(
            COUNT * sizeof(long double))) {
        std::cerr << "Unexpected reference file size\n";
        return 1;
    }

    long double min_value = std::numeric_limits<long double>::max();
    long double max_value = -std::numeric_limits<long double>::max();
    size_t non_finite = 0;

    for (long double v : values) {
        if (!std::isfinite(v)) {
            ++non_finite;
            continue;
        }
        if (v < min_value) min_value = v;
        if (v > max_value) max_value = v;
    }

    std::cout << std::setprecision(18);
    std::cout << "Logit count: " << values.size() << '\n';
    std::cout << "Non-finite values: " << non_finite << '\n';
    std::cout << "Minimum finite logit: " << min_value << '\n';
    std::cout << "Maximum finite logit: " << max_value << '\n';
    return non_finite == 0 ? 0 : 2;
}
