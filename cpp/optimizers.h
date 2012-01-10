#ifndef optimizers_H
#define optimizers_H
#include "globals.h"
#include "predictor.h"

void sgd(Predictor& p, Data *ratings, int num_ratings);
void gd(Predictor& p, Data *ratings, int num_ratings);
double compute_gradient(Predictor& p, Data *ratings, int num_ratings, float *movie_gradient, float *user_gradient);
void set_defaults();
#endif
