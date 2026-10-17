#include <iomanip>
#include <iostream>
#include <sw/universal/number/dd/dd.hpp>

int main() {
    using sw::universal::dd;

    dd a = dd(1.0e16);
    dd b = dd(1.0);
    dd dd_result = (a + b) - a;

    double da = 1.0e16;
    double db = 1.0;
    double double_result = (da + db) - da;

    std::cout << std::setprecision(17);
    std::cout << "dd result, converted to double: "
              << static_cast<double>(dd_result) << '\n';
    std::cout << "double result:                  "
              << double_result << '\n';
}
