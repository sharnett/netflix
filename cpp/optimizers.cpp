#include "optimizers.h"

using namespace std;

static time_t start, end;

void sgd(Predictor& p, Data *ratings, int num_ratings, Data *cv_ratings, int num_cv_ratings, Settings s) {
    cout << "doing stochastic gradient descent" << endl;
    int e, i, user, cnt = 0, num_f=p.get_num_features();
    Data* rating;
    double err, sq, rmse_last=3, rmse = 2.0, alpha = s.lrate, decay=.98;
    short movie;
    float *uf, *mf, *temp;
    temp = new float[num_f];

    time(&start);
    for (e=0; (e < s.min_epochs) || (rmse <= rmse_last - s.min_improvement); e++) {
        if (e == s.max_epochs) break;
        cnt++;
        sq = 0;
        rmse_last = rmse;

        for (i=0; i<num_ratings; i++) {
            rating = ratings + i;
            movie = rating->movie;
            user = rating->user;

            err = p.predict(user, movie) - (double)rating->rating;
            sq += err*err;
            
            uf = &p.user_features[user*num_f];
            mf = &p.movie_features[movie*num_f];
            cblas_scopy(num_f, uf, 1, temp, 1);
            cblas_saxpy(num_f, err/s.K, mf, 1, temp, 1);
            cblas_saxpy(num_f, -s.K*alpha, temp, 1, 
                    &p.user_features[user*num_f], 1);
            cblas_scopy(num_f, mf, 1, temp, 1);
            cblas_saxpy(num_f, err/s.K, uf, 1, temp, 1);
            cblas_saxpy(num_f, -s.K*alpha, temp, 1, 
                    &p.movie_features[movie*num_f], 1);
        }
        rmse = sqrt(sq/num_ratings);
        //rmse = cost(p, cv_ratings, num_cv_ratings);
        alpha *= decay;
        time(&end);
        cout << cnt << " " << rmse << " time: " << difftime(end,start) << "s" << endl;
    }
    if (e >= s.max_epochs) 
        cout << "stochastic gradient descent finished with max iterations: " << e << endl;
    else if (rmse > rmse_last - s.min_improvement) 
        cout << "stochastic gradient descent finished. Improvement " <<
            rmse_last-rmse << " is less than minimum " << s.min_improvement << endl;
    delete [] temp;
}

//void gd(Predictor& p, Data *ratings, int num_ratings, Settings s) {
//    cout << "doing gradient descent" << endl;
//    int e, cnt = 0, num_f=p.get_num_features(),
//        num_m = p.get_num_movies(), num_u = p.get_num_users(); 
//    float sq, rmse_last=3, rmse = 2.0;
//    float *movie_gradient = new float[(num_m+num_u)*num_f];
//    float *user_gradient = movie_gradient + (num_m*num_f);
//    float *uf = p.user_features, *mf = p.movie_features;
//
//    //time_t start,end;
//    for (e=0; (e < s.min_epochs) || (rmse <= rmse_last - s.min_improvement); e++) {
//        time(&start);
//        if (e == s.max_epochs) break;
//        cnt++;
//        rmse_last = rmse;
//
//        sq = compute_gradient(p, ratings, num_ratings, movie_gradient, 
//                user_gradient, s.K);
//        cblas_saxpy(num_m*num_f, -s.lrate/(1-s.lrate*s.K), movie_gradient, 1, mf, 1); 
//        cblas_sscal(num_m*num_f, 1-s.lrate*s.K, mf, 1); 
//        cblas_saxpy(num_u*num_f, -s.lrate/(1-s.lrate*s.K), user_gradient, 1, uf, 1); 
//        cblas_sscal(num_u*num_f, 1-s.lrate*s.K, uf, 1); 
//        rmse = sqrt(sq/num_ratings);
//        time(&end);
//        cout << cnt << " " << rmse << " time: " << difftime(end,start) << "s" << endl;
//    }
//}

float compute_gradient(Predictor& p, Data *ratings, int num_ratings, Data *cv_ratings, int num_cv_ratings, 
        float *movie_gradient, float *user_gradient, float K) {
    int nf = p.get_num_features(), nm = p.get_num_movies(), nu = p.get_num_users(); 
    float sq=0;
    //cblas_sscal((nm+nu)*nf, 0, movie_gradient, 1);
    //memset(movie_gradient, 0, sizeof(float) * (nm+nu)*nf);
    for (int i=0; i<(nm+nu)*nf; i++) movie_gradient[i] = 0;
    #pragma omp parallel
    {
        Data *rating;
        int user, movie;
        double err, lcl_sq=0;
        float *user_features, *movie_features, *lcl_movie_grad, *lcl_user_grad;
        lcl_movie_grad = new float[(nm+nu)*nf] ();
        lcl_user_grad = lcl_movie_grad + (nm*nf);

        #pragma omp for
        for (int i=0; i<num_ratings; i++) {
            rating = ratings + i;
            user = rating->user;
            movie = rating->movie;

            err = p.predict(user, movie) - (double)rating->rating;
            lcl_sq += err*err;

            user_features = &p.user_features[user*nf];
            movie_features = &p.movie_features[movie*nf];
            cblas_saxpy(nf, K, user_features, 1, &lcl_user_grad[user*nf], 1);
            cblas_saxpy(nf, err, movie_features, 1, &lcl_user_grad[user*nf], 1);
            cblas_saxpy(nf, K, movie_features, 1, &lcl_movie_grad[movie*nf], 1);
            cblas_saxpy(nf, err, user_features, 1, &lcl_movie_grad[movie*nf], 1);
        }
        #pragma omp critical
        {
            sq += lcl_sq;
            cblas_saxpy((nm+nu)*nf, (float)10/num_ratings, lcl_movie_grad, 1,
                    movie_gradient, 1);
        }
        delete [] lcl_movie_grad;
    }
    cout << "." << flush;
    return sqrt(sq/num_ratings);
    //return cost(p, cv_ratings, num_cv_ratings);
}

void bfgs(Predictor& p, Data *ratings, int num_ratings, Data *cv_ratings, int num_cv_ratings, Settings s) { 
    int num_f=p.get_num_features(), num_m = p.get_num_movies(), num_u = p.get_num_users(); 
    float *movie_gradient = new float[(num_m+num_u)*num_f];
    float *user_gradient = movie_gradient + (num_m*num_f);
    int n = p.get_num_features() * (p.get_num_movies() + p.get_num_users());
    real_1d_array x; x.setlength(n);
    for (int i=0; i<n; i++) x[i] = p.movie_features[i];
    float epsg = 0;
    float epsf = s.min_improvement;
    float epsx = 0;
    ae_int_t maxits = s.max_epochs;
    mincgstate state;
    mincgreport rep;
    BFGS_ptr b(p, ratings, num_ratings, cv_ratings, num_cv_ratings, movie_gradient, user_gradient, s);

    mincgcreate(n, x, state);
    mincgsetcond(state, epsg, epsf, epsx, maxits);
    mincgsetxrep(state, true);
    cout << "Optimizing.." << flush;
    time(&start);
    alglib::mincgoptimize(state, bfgs_grad, bfgs_callback, &b);
    mincgresults(state, x, rep);
    cout << "\nOptimization complete." << endl;

    cout << rep.iterationscount << " iterations, " << rep.nfev << " function evaluations" << endl;
    switch(rep.terminationtype) {
        case -2:
            cout << "rounding errors prevent further improvement. X contains "
                    "best point found." << endl;
            break;
        case -1:
            cout << "incorrect parameters were specified" << endl;
            break;
        case 1:
            cout << "success. relative function improvement is no more than " << epsf << endl;
            break;
        case 2:
            cout << "success. relative step size is no more than " << epsx << endl;
            break;
        case 4:
            cout << "success. gradient norm is no more than " << epsg << endl;
            break;
        case 5:
            cout << "maximum number of iterations reached" << endl;
            break;
        case 7:
            cout << "stopping conditions are too stringent, further improvement "
                    "is impossible" << endl;
            break;
    }
}

void bfgs_grad(const real_1d_array &x, double &f, real_1d_array &grad, void *p) {
    BFGS_ptr *b = (BFGS_ptr *)p;
    int n = x.length();
    for (int i=0; i<n; i++)
        b->predictor.movie_features[i] = x[i];
    f = compute_gradient(b->predictor, b->ratings, b->num_ratings, b->cv_ratings, b->num_cv_ratings,
            b->movie_gradient, b->user_gradient, b->settings.K);
    for (int i=0; i<n; i++)
        grad[i] = b->movie_gradient[i];
}

void bfgs_callback(const real_1d_array &x, double f, void *p) {
    static int i = 0;
    i++;
    time(&end); 
    printf("\n%3d %9.6f %5ds ", i, f, (int)difftime(end, start));
    cout << flush;
    BFGS_ptr *b = (BFGS_ptr *)p;
    cout << "cv: " << cost(b->predictor, b->cv_ratings, b->num_cv_ratings);
}

double cost(Predictor& p, Data *ratings, int num_ratings) {
    double sq = 0;
    #pragma omp parallel reduction(+: sq)
    {
        double err, lcl_sq = 0;
        int user, movie;
        Data *rating;
        #pragma omp for
        for (int i=0; i<num_ratings; i++) {
            rating = ratings + i;
            movie = rating->movie;
            user = rating->user;
            err = p.predict(user, movie) - (double)rating->rating;
            lcl_sq += err*err;
        }
        sq += lcl_sq;
    }
    return sqrt(sq/num_ratings);
}

// non-BLAS versions of linear algebra
// aka non-vectorized loops
//
// void sgd(Predictor& p, Data *ratings, int num_ratings, Settings s) {
// ...
//            for (f=0; f<num_f; f++) {
//                cf = p.user_features[user][f];
//                mf = p.movie_features[movie][f];
//                p.user_features[user][f] += (double)(LRATE * (err * mf - K * cf));
//                p.movie_features[movie][f] += (double)(LRATE * (err * cf - K * mf));
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
//            user_gradient[user][f] += -1*(double) (err * mf - K * cf);
//            movie_gradient[movie][f] += -1*(double) (err * cf - K * mf);
//        }
