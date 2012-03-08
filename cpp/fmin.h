#ifndef fmin_h
#define fmin_h

#include <iostream>
#include <cmath>
#include "stdlib.h"

using namespace std;

double local_min_rc(double &a, double &b, int &status, double value);
double sign(double x);
bool fminbnd(double (*f)(double), double a, double b, double& min, int& num_iterations, bool verbose);
double fminbnd(double (*f)(double), double a, double b);

#endif
