#include <iostream>
#include <iomanip>
#include <cmath>
#include <universal/number/cfloat/cfloat.hpp>
#include <universal/number/posit/posit.hpp>

int main() {
    using namespace sw::universal;

    using FP32 = cfloat<32, 8>;
    using Posit32 = posit<32, 2>;

    const double reference = 3.14159265358979323846 * 2.0;

    FP32 fp_a = 3.14159265358979323846;
    FP32 fp_b = 2.0;
    FP32 fp_result = fp_a * fp_b;

    Posit32 p_a = 3.14159265358979323846;
    Posit32 p_b = 2.0;
    Posit32 p_result = p_a * p_b;

    double fp_value = static_cast<double>(fp_result);
    double posit_value = static_cast<double>(p_result);

    std::cout << std::setprecision(15);
    std::cout << "Reference:      " << reference << '\n';
    std::cout << "FP32 result:    " << fp_value << '\n';
    std::cout << "FP32 abs error: " << std::abs(fp_value - reference) << '\n';
    std::cout << "Posit32 result: " << posit_value << '\n';
    std::cout << "Posit32 error:  " << std::abs(posit_value - reference) << '\n';

    return 0;
}
