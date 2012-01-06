#ifndef optimizers_H
#define optimizers_H
#include "globals.h"
#include "predictor.h"

void sgd(Predictor& p, Data *ratings, int num_ratings);
void gd(Predictor& p, Data *ratings, int num_ratings);
double compute_gradient(Data *ratings, int num_ratings, float *movie_gradient[MAX_MOVIES], float *user_gradient[MAX_USERS], Predictor& p);
void set_defaults();
#endif
