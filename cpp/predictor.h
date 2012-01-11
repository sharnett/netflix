#ifndef predictor_H
#define predictor_H

#include <Accelerate/Accelerate.h> // Mac OS X specific, has BLAS headers
#include "load.h"

class Predictor {
public:
    Predictor(int num_users, int num_movies, int nf);
    ~Predictor();
    double predict(int user, short movie);
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
    float *movie_avg;
};

// worthless random number generator helper
float rndn();

#endif
