#include <iostream>
#include <limits>
#include <cfloat>

int main() {
    std::cout << "sizeof(float): " << sizeof(float) << '\n';
    std::cout << "sizeof(double): " << sizeof(double) << '\n';
    std::cout << "sizeof(long double): " << sizeof(long double) << '\n';

    std::cout << "LDBL_MANT_DIG: " << LDBL_MANT_DIG << '\n';
    std::cout << "LDBL_EPSILON: " << std::numeric_limits<long double>::epsilon() << '\n';
    std::cout << "LDBL_MAX_DIGITS10: " << std::numeric_limits<long double>::digits10 << '\n';
}
