#include "optimizers.h"
#include <cmath>
#include <iostream>

using namespace std;

double MIN_IMPROVEMENT;        // Minimum improvement required to continue current feature
int MIN_EPOCHS;           // Minimum number of epochs per feature
int MAX_EPOCHS;           // Max epochs per feature
double INIT;           // Initialization value for features
double LRATE;         // Learning rate parameter
double K;         // Regularization parameter used to minimize over-fitting

void sgd(Predictor& p, Data *ratings, int num_ratings) {
    cout << "doing stochastic gradient descent" << endl;
    int f, e, i, user, cnt = 0, MAX_FEATURES=p.get_num_features();
    Data* rating;
    double err, prediction, sq, rmse_last, rmse = 2.0;
    short movie;
    float cf, mf;

    for (e=0; (e < MIN_EPOCHS) || (rmse <= rmse_last - MIN_IMPROVEMENT); e++) {
        if (e > MAX_EPOCHS) break;
        cnt++;
        sq = 0;
        rmse_last = rmse;

        for (i=0; i<num_ratings; i++) {
            rating = ratings + i;
            movie = rating->movie;
            user = rating->user;

            prediction = p.predict(user, movie);
            err = (1.0 * rating->rating - prediction);
            sq += err*err;
            
            for (f=0; f<MAX_FEATURES; f++) {
                cf = p.user_features[user][f];
                mf = p.movie_features[movie][f];
                p.user_features[user][f] += (float)(LRATE * (err * mf - K * cf));
                p.movie_features[movie][f] += (float)(LRATE * (err * cf - K * mf));
            }
        }
        rmse = sqrt(sq/num_ratings);
        cout << cnt << " " << rmse << endl;
    }
}

void gd(Predictor& p, Data *ratings, int num_ratings) {
    cout << "doing gradient descent" << endl;
    int f, e, cnt = 0, MAX_FEATURES=p.get_num_features();
    double sq, rmse_last, rmse = 2.0;
    short movie; int user;
    float *movie_gradient[MAX_MOVIES];
    float *user_gradient[MAX_USERS];
    for (movie=0; movie<MAX_MOVIES; movie++) movie_gradient[movie] = new float[MAX_FEATURES];
    for (user=0; user<MAX_USERS; user++) user_gradient[user] = new float[MAX_FEATURES];

    for (e=0; (e < MIN_EPOCHS) || (rmse <= rmse_last - MIN_IMPROVEMENT); e++) {
        if (e > MAX_EPOCHS) break;
        cnt++;
        rmse_last = rmse;

        sq = compute_gradient(ratings, num_ratings, movie_gradient, user_gradient, p);
        for (f=0; f<MAX_FEATURES; f++) {
            for (movie=0; movie<MAX_MOVIES; movie++) 
                p.movie_features[movie][f] -= LRATE * movie_gradient[movie][f];
            for (user=0; user<MAX_USERS; user++) 
                p.user_features[user][f] -= LRATE * user_gradient[user][f];
        }
        rmse = sqrt(sq/num_ratings);
        cout << cnt << " " << rmse << endl;
    }
}

double compute_gradient(Data *ratings, int num_ratings, float *movie_gradient[MAX_MOVIES], float *user_gradient[MAX_USERS], Predictor& p) {
    int user, f, MAX_FEATURES=p.get_num_features();
    short movie;
    double err, prediction, sq=0; 
    float cf, mf;
    Data *rating;
    for (f=0; f<MAX_FEATURES; f++) {
        for (movie=0; movie<MAX_MOVIES; movie++) movie_gradient[movie][f] = 0;
        for (user=0; user<MAX_USERS; user++) user_gradient[user][f] = 0;
    }
    for (int i=0; i<num_ratings; i++) {
        rating = ratings + i;
        movie = rating->movie;
        user = rating->user;

        prediction = p.predict(user, movie);
        err = (1.0 * rating->rating - prediction);
        sq += err*err;
        for (f=0; f<MAX_FEATURES; f++) {
            cf = p.user_features[user][f];
            mf = p.movie_features[movie][f];
            user_gradient[user][f] += -1*(float) (err * mf - K * cf);
            movie_gradient[movie][f] += -1*(float) (err * cf - K * mf);
        }
    }
    return sq;
}

void set_defaults() {
    MIN_IMPROVEMENT = 0.001;        // Minimum improvement required to continue current feature
    MIN_EPOCHS = 5;           // Minimum number of epochs per feature
    MAX_EPOCHS = 20;           // Max epochs per feature
    INIT = 0.1;           // Initialization value for features
    LRATE = 0.001;         // Learning rate parameter
    K = 0.015;         // Regularization parameter used to minimize over-fitting
}
