#include <iomanip>
#include <iostream>
#include <sw/universal/number/dd/dd.hpp>

int main() {
    using sw::universal::dd;

    dd one = dd(1.0);
    dd three = dd(3.0);
    dd result = one / three;

    double converted = static_cast<double>(result);

    std::cout << std::setprecision(17);
    std::cout << "dd(1/3) converted to double: "
              << converted << '\n';
    std::cout << "double 1/3:                  "
              << (1.0 / 3.0) << '\n';
    std::cout << "dd arithmetic test completed\n";
}
