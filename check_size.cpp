#include <iostream>
#include<limits>
using namespace std;

int main() {
    cout << "float: " << sizeof(float) * 8 << " bits\n";
    cout << "double: " << sizeof(double) * 8 << " bits\n";
    cout << "long double: " << sizeof(long double) * 8 << " bits\n";
    cout << "long double decimal digits: "
         << numeric_limits<long double>::digits10 << "\n";
}
