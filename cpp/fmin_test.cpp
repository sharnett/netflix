#include <iostream>
#include "fmin.h"

double f(double x) { return x*(x-1); }

int main() {
    cout << fminbnd(f, -100, 100) << endl;
//    double min;
//    int num_iterations;
//    if (!fminbnd(f, -10000, 10000, min, num_iterations))
//        cout << "failed to converge";
//    else
//        cout << "converged to " << min;
//    cout << " after " << num_iterations << " iterations" << endl;
    return 0;
}
