#include <iostream>
#include <sw/universal/number/posit/posit.hpp>

int main() {
    sw::universal::posit<32, 2> a = 1.5;
    sw::universal::posit<32, 2> b = 2.25;
    sw::universal::posit<32, 2> c = a * b;

    std::cout << "a = " << a << '\n';
    std::cout << "b = " << b << '\n';
    std::cout << "a * b = " << c << '\n';

    return 0;
}
