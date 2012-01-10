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
    int e, i, user, cnt = 0, num_f=p.get_num_features();
    Data* rating;
    double err, prediction, sq, rmse_last, rmse = 2.0;
    short movie;
//    float cf, mf;
    float *cf, *mf, *temp;
    temp = new float[num_f];

    time_t start,end;
    for (e=0; (e < MIN_EPOCHS) || (rmse <= rmse_last - MIN_IMPROVEMENT); e++) {
        time(&start);
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
            
            // BLAS is ~25% faster than naive method
            cf = &p.user_features[user*num_f];
            mf = &p.movie_features[movie*num_f];
            cblas_scopy(num_f, cf, 1, temp, 1);
            cblas_saxpy(num_f, -err/K, mf, 1, temp, 1);
            cblas_saxpy(num_f, -K*LRATE, temp, 1, 
                    &p.user_features[user*num_f], 1);
            cblas_scopy(num_f, mf, 1, temp, 1);
            cblas_saxpy(num_f, -err/K, cf, 1, temp, 1);
            cblas_saxpy(num_f, -K*LRATE, temp, 1, 
                    &p.movie_features[movie*num_f], 1);
//            for (f=0; f<num_f; f++) {
//                cf = p.user_features[user][f];
//                mf = p.movie_features[movie][f];
//                p.user_features[user][f] += (float)(LRATE * (err * mf - K * cf));
//                p.movie_features[movie][f] += (float)(LRATE * (err * cf - K * mf));
//            }
        }
        rmse = sqrt(sq/num_ratings);
        time(&end);
        cout << cnt << " " << rmse << " time: " << difftime(end,start) << "s" << endl;
    }
    delete [] temp;
}

void gd(Predictor& p, Data *ratings, int num_ratings) {
    cout << "doing gradient descent" << endl;
    int e, cnt = 0, num_f=p.get_num_features(),
        num_m = p.get_num_movies(), num_u = p.get_num_users(); 
    double sq, rmse_last, rmse = 2.0;
    float *movie_gradient = new float[num_m*num_f];
    float *user_gradient = new float[num_u*num_f];
    float *uf = p.user_features, *mf = p.movie_features;

    time_t start,end;
    for (e=0; (e < MIN_EPOCHS) || (rmse <= rmse_last - MIN_IMPROVEMENT); e++) {
        time(&start);
        if (e > MAX_EPOCHS) break;
        cnt++;
        rmse_last = rmse;

        sq = compute_gradient(p, ratings, num_ratings, movie_gradient, user_gradient);
        // should BLAS this out
        // and add regularization, jerk off
//        for (f=0; f<num_f; f++) {
//            for (movie=0; movie<num_m; movie++) 
//                p.movie_features[movie][f] -= LRATE * movie_gradient[movie][f];
//            for (user=0; user<MAX_USERS; user++) 
//                p.user_features[user][f] -= LRATE * user_gradient[user][f];
//        }
        cblas_saxpy(num_m*num_f, -LRATE/(1-LRATE*K), movie_gradient, 1, mf, 1); 
        cblas_saxpy(num_u*num_f, -LRATE/(1-LRATE*K), user_gradient, 1, uf, 1); 
        rmse = sqrt(sq/num_ratings);
        time(&end);
        cout << cnt << " " << rmse << " time: " << difftime(end,start) << "s" << endl;
    }
}

double compute_gradient(Predictor& p, Data *ratings, int num_ratings, 
        float *movie_gradient, float *user_gradient) {
    int user, f, num_f = p.get_num_features(), 
        num_m = p.get_num_movies(), num_u = p.get_num_users(); 
    short movie;
    double err, prediction, sq=0; 
    float *cf, *mf, *temp;
    temp = new float[num_f];
    Data *rating;
    for (int i=0; i<num_m*num_f; i++) movie_gradient[i] = 0;
    for (int i=0; i<num_u*num_f; i++) user_gradient[i] = 0;

    for (int i=0; i<num_ratings; i++) {
        rating = ratings + i;
        movie = rating->movie;
        user = rating->user;

        prediction = p.predict(user, movie);
        err = (1.0 * rating->rating - prediction);
        sq += err*err;
        // BLAS is ~25% faster than naive method
        cf = &p.user_features[user*num_f];
        mf = &p.movie_features[movie*num_f];
        cblas_scopy(num_f, cf, 1, temp, 1);
        cblas_saxpy(num_f, -err/K, mf, 1, temp, 1);
        cblas_saxpy(num_f, K, temp, 1, &user_gradient[user*num_f], 1);
        cblas_scopy(num_f, mf, 1, temp, 1);
        cblas_saxpy(num_f, -err/K, cf, 1, temp, 1);
        cblas_saxpy(num_f, K, temp, 1, &movie_gradient[movie*num_f], 1);
//        for (f=0; f<num_f; f++) {
//            cf = p.user_features[user][f];
//            mf = p.movie_features[movie][f];
//            user_gradient[user][f] += -1*(float) (err * mf - K * cf);
//            movie_gradient[movie][f] += -1*(float) (err * cf - K * mf);
//        }
    }
    delete [] temp;
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
