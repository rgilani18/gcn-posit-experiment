#include <iostream>
#include <universal/number/posit/posit.hpp>

int main() {
    using namespace sw::universal;

    posit<32, 2> a = 3.141592653589793;
    posit<32, 2> b = 2.0;
    posit<32, 2> c = a * b;

    std::cout << "Posit32 arithmetic test\n";
    std::cout << "a = " << a << '\n';
    std::cout << "b = " << b << '\n';
    std::cout << "a * b = " << c << '\n';

    return 0;
}
