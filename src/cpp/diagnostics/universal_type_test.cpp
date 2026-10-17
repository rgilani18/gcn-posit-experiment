#include <iostream>
#include <iomanip>

#include <universal/number/cfloat/cfloat.hpp>
#include <universal/number/posit/posit.hpp>

int main() {
    using FP32 = sw::universal::fp32;
    using Posit32 = sw::universal::posit<32, 2>;

    FP32 f1 = 1.5;
    FP32 f2 = 2.25;

    Posit32 p1 = 1.5;
    Posit32 p2 = 2.25;

    FP32 fp_result = f1 * f2;
    Posit32 posit_result = p1 * p2;

    std::cout << std::setprecision(12);
    std::cout << "Universal FP32 result: " << fp_result << '\n';
    std::cout << "Universal Posit32 result: " << posit_result << '\n';

    return 0;
}
