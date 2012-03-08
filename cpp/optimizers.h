#ifndef optimizers_H
#define optimizers_H
#include <cmath>
#include "globals.h"
#include "predictor.h"
#include "stdafx.h"
#include "optimization.h"

struct Settings {
    Settings(): num_features(4), min_improvement(.001), min_epochs(5), 
        max_epochs(20), lrate(.001), K(.015), dump(0) {}

    int num_features;
    double min_improvement; // min improvement required to continue iterating
    int min_epochs;         // min number of 'epochs', aka sweeps through entire
    int max_epochs;         //     training set
    double lrate;           // learning rate parameter
    double K;               // regularization parameter to control over-fitting
    bool dump;              // 1 if you want to dump to features.bin after 
};

struct BFGS_ptr {
    BFGS_ptr(Predictor& p, Data *r, int n, float *mg, float *ug, Settings s):
            predictor(p), ratings(r), num_ratings(n), movie_gradient(mg),
            user_gradient(ug), settings(s) {}

    Predictor& predictor;
    Data *ratings;
    int num_ratings;
    float *movie_gradient;
    float *user_gradient;
    Settings settings;
};

void sgd(Predictor& p, Data *ratings, int num_ratings, Settings s);
void gd(Predictor& p, Data *ratings, int num_ratings, Settings s);
float compute_gradient(Predictor& p, Data *ratings, int num_ratings, 
        float *movie_gradient, float *user_gradient, float K);

void bfgs(Predictor& p, Data *ratings, int num_ratings, Settings s);
using namespace alglib;
void bfgs_grad(const real_1d_array &x, double &f, real_1d_array &grad, void *p);
void bfgs_callback(const real_1d_array &x, double f, void *p);

#endif
