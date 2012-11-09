#ifndef predictor_H
#define predictor_H

#if defined __APPLE__
#include <Accelerate/Accelerate.h> // Mac OS X specific, has BLAS headers
#elif defined __linux__
#include <cblas.h> //for linux
//extern "C" {
//#include "/home/sean/lib/atlas/include/cblas.h"
//}
#endif

#include "load.h"
#include <string>

using namespace std;

class Predictor {
public:
    Predictor(int num_users, int num_movies, int nf);
    ~Predictor();
    float predict(int user, short movie);
    int get_num_users() { return num_users; }
    int get_num_movies() { return num_movies; }
    int get_num_features() { return num_features; }

    // the real meat of the predictor
    float *movie_features;
    float *user_features;

private:
    int num_users;
    int num_movies;
    int num_features;

    double *movie_avg;
    double *user_avg;
    double average;
};

// worthless random number generator helper
double rndn();

#endif
