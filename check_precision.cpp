#include <iostream>
#include <limits>

int main() {
    std::cout << "float bits:       "
              << sizeof(float) * 8 << "\n";

    std::cout << "double bits:      "
              << sizeof(double) * 8 << "\n";

    std::cout << "long double bits: "
              << sizeof(long double) * 8 << "\n";

    std::cout << "float digits:       "
              << std::numeric_limits<float>::digits10 << "\n";

    std::cout << "double digits:      "
              << std::numeric_limits<double>::digits10 << "\n";

    std::cout << "long double digits: "
              << std::numeric_limits<long double>::digits10 << "\n";
}
