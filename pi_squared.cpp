#include <iostream>
#include <iomanip>
#include <cmath>

#include <universal/number/posit/posit.hpp>

int main() {
    using Posit32 = sw::universal::posit<32, 2>;

    const long double pi = acosl(-1.0L);

    // High-precision reference
    const long double reference = pi * pi;

    // FP32
    const float pi_fp32 = static_cast<float>(pi);
    const float fp32_result = pi_fp32 * pi_fp32;

    // Posit32
    const Posit32 pi_posit = Posit32(pi);
    const Posit32 posit_result = pi_posit * pi_posit;

    long double posit_as_long_double =
        static_cast<long double>(posit_result);

    long double fp32_error =
        fabsl(static_cast<long double>(fp32_result) - reference);

    long double posit_error =
        fabsl(posit_as_long_double - reference);

    std::cout << std::setprecision(20);

    std::cout << "Reference pi^2 : "
              << reference << '\n';

    std::cout << "FP32 pi^2      : "
              << fp32_result << '\n';

    std::cout << "Posit32 pi^2    : "
              << posit_as_long_double << '\n';

    std::cout << '\n';

    std::cout << "FP32 abs error  : "
              << fp32_error << '\n';

    std::cout << "Posit abs error  : "
              << posit_error << '\n';

    return 0;
}
