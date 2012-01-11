#include "optimizers.h"

using namespace std;

void sgd(Predictor& p, Data *ratings, int num_ratings, Settings s) {
    cout << "doing stochastic gradient descent" << endl;
    int e, i, user, cnt = 0, num_f=p.get_num_features();
    Data* rating;
    double err, prediction, sq, rmse_last, rmse = 2.0;
    short movie;
    float *uf, *mf, *temp;
    temp = new float[num_f];

    time_t start,end;
    for (e=0; (e < s.min_epochs) || (rmse <= rmse_last - s.min_improvement); e++) {
        time(&start);
        if (e == s.max_epochs) break;
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
            
            uf = &p.user_features[user*num_f];
            mf = &p.movie_features[movie*num_f];
            cblas_scopy(num_f, uf, 1, temp, 1);
            cblas_saxpy(num_f, -err/s.K, mf, 1, temp, 1);
            cblas_saxpy(num_f, -s.K*s.lrate, temp, 1, 
                    &p.user_features[user*num_f], 1);
            cblas_scopy(num_f, mf, 1, temp, 1);
            cblas_saxpy(num_f, -err/s.K, uf, 1, temp, 1);
            cblas_saxpy(num_f, -s.K*s.lrate, temp, 1, 
                    &p.movie_features[movie*num_f], 1);
        }
        rmse = sqrt(sq/num_ratings);
        time(&end);
        cout << cnt << " " << rmse << " time: " << difftime(end,start) << "s" << endl;
    }
    delete [] temp;
}

void gd(Predictor& p, Data *ratings, int num_ratings, Settings s) {
    cout << "doing gradient descent" << endl;
    int e, cnt = 0, num_f=p.get_num_features(),
        num_m = p.get_num_movies(), num_u = p.get_num_users(); 
    double sq, rmse_last, rmse = 2.0;
    float *movie_gradient = new float[num_m*num_f];
    float *user_gradient = new float[num_u*num_f];
    float *uf = p.user_features, *mf = p.movie_features;

    time_t start,end;
    for (e=0; (e < s.min_epochs) || (rmse <= rmse_last - s.min_improvement); e++) {
        time(&start);
        if (e == s.max_epochs) break;
        cnt++;
        rmse_last = rmse;

        sq = compute_gradient(p, ratings, num_ratings, movie_gradient, 
                user_gradient, s);
        cblas_saxpy(num_m*num_f, -s.lrate/(1-s.lrate*s.K), movie_gradient, 1, mf, 1); 
        cblas_sscal(num_m*num_f, 1-s.lrate*s.K, mf, 1); 
        cblas_saxpy(num_u*num_f, -s.lrate/(1-s.lrate*s.K), user_gradient, 1, uf, 1); 
        cblas_sscal(num_u*num_f, 1-s.lrate*s.K, uf, 1); 
        rmse = sqrt(sq/num_ratings);
        time(&end);
        cout << cnt << " " << rmse << " time: " << difftime(end,start) << "s" << endl;
    }
}

double compute_gradient(Predictor& p, Data *ratings, int num_ratings, 
        float *movie_gradient, float *user_gradient, Settings s) {
    int user, num_f = p.get_num_features(), 
        num_m = p.get_num_movies(), num_u = p.get_num_users(); 
    short movie;
    double err, prediction, sq=0; 
    float *uf, *mf, *temp;
    temp = new float[num_f];
    Data *rating;
    for (int i=0; i<num_m*num_f; i++) movie_gradient[i] = 0;
    for (int i=0; i<num_u*num_f; i++) user_gradient[i] = 0;

    for (int i=0; i<num_ratings; i++) {
        rating = ratings + i;
        user = rating->user;
        movie = rating->movie;

        prediction = p.predict(user, movie);
        err = (1.0 * rating->rating - prediction);
        sq += err*err;

        uf = &p.user_features[user*num_f];
        mf = &p.movie_features[movie*num_f];
        cblas_scopy(num_f, uf, 1, temp, 1);
        cblas_saxpy(num_f, -err/s.K, mf, 1, temp, 1);
        cblas_saxpy(num_f, s.K, temp, 1, &user_gradient[user*num_f], 1);
        cblas_scopy(num_f, mf, 1, temp, 1);
        cblas_saxpy(num_f, -err/s.K, uf, 1, temp, 1);
        cblas_saxpy(num_f, s.K, temp, 1, &movie_gradient[movie*num_f], 1);
    }
    delete [] temp;
    return sq;
}

// non-BLAS versions of linear algebra
// aka non-vectorized loops
//
// void sgd(Predictor& p, Data *ratings, int num_ratings, Settings s) {
// ...
//            for (f=0; f<num_f; f++) {
//                cf = p.user_features[user][f];
//                mf = p.movie_features[movie][f];
//                p.user_features[user][f] += (float)(LRATE * (err * mf - K * cf));
//                p.movie_features[movie][f] += (float)(LRATE * (err * cf - K * mf));
//            }
//
// void gd(Predictor& p, Data *ratings, int num_ratings, Settings s) {
// ...
//        for (f=0; f<num_f; f++) {
//            for (movie=0; movie<num_m; movie++) 
//                p.movie_features[movie][f] -= LRATE * movie_gradient[movie][f];
//            for (user=0; user<MAX_USERS; user++) 
//                p.user_features[user][f] -= LRATE * user_gradient[user][f];
//        }
//
// double compute_gradient(Predictor& p, Data *ratings, int num_ratings, 
// ...
//        for (f=0; f<num_f; f++) {
//            cf = p.user_features[user][f];
//            mf = p.movie_features[movie][f];
//            user_gradient[user][f] += -1*(float) (err * mf - K * cf);
//            movie_gradient[movie][f] += -1*(float) (err * cf - K * mf);
//        }
