#ifndef optimizers_H
#define optimizers_H
#include <cmath>
#include "globals.h"
#include "predictor.h"

struct Settings {
    Settings(): num_features(4), min_improvement(.001), min_epochs(5), 
        max_epochs(20), lrate(.001), K(.015) {}

    int num_features;
    double min_improvement; // min improvement required to continue iterating
    int min_epochs;         // min number of 'epochs', aka sweeps through entire
    int max_epochs;         // training set
    double lrate;           // learning rate parameter
    double K;               // regularization parameter to control over-fitting
};

void sgd(Predictor& p, Data *ratings, int num_ratings, Settings s);
void gd(Predictor& p, Data *ratings, int num_ratings, Settings s);
double compute_gradient(Predictor& p, Data *ratings, int num_ratings, 
        float *movie_gradient, float *user_gradient, Settings s);
#endif
