#include "optimizers.h"

using namespace std;

void sgd(Predictor& p, Data *ratings, int num_ratings, Settings s) {
    cout << "doing stochastic gradient descent" << endl;
    int e, i, user, cnt = 0, num_f=p.get_num_features();
    Data* rating;
    double err, prediction, sq, rmse_last, rmse = 2.0;
    short movie;
    double *uf, *mf, *temp;
    temp = new double[num_f];

    time_t start,end;
    for (e=0; (e < s.min_epochs) || (rmse <= rmse_last - s.min_improvement); e++) {
        time(&start);
        if (e == s.max_epochs) break;
        cnt++;
        sq = 0;
        rmse_last = rmse;

//        #pragma omp parallel
//        {
//        #pragma omp for
        for (i=0; i<num_ratings; i++) {
            rating = ratings + i;
            movie = rating->movie;
            user = rating->user;

            prediction = p.predict(user, movie);
            err = (1.0 * rating->rating - prediction);
            sq += err*err;
            
            uf = &p.user_features[user*num_f];
            mf = &p.movie_features[movie*num_f];
            cblas_dcopy(num_f, uf, 1, temp, 1);
            cblas_daxpy(num_f, -err/s.K, mf, 1, temp, 1);
            cblas_daxpy(num_f, -s.K*s.lrate, temp, 1, 
                    &p.user_features[user*num_f], 1);
            cblas_dcopy(num_f, mf, 1, temp, 1);
            cblas_daxpy(num_f, -err/s.K, uf, 1, temp, 1);
            cblas_daxpy(num_f, -s.K*s.lrate, temp, 1, 
                    &p.movie_features[movie*num_f], 1);
        }
//        }
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
//    double *movie_gradient = new float[num_m*num_f];
//    double *user_gradient = new float[num_u*num_f];
    double *movie_gradient = new double[(num_m+num_u)*num_f];
    double *user_gradient = movie_gradient + (num_m*num_f);
    double *uf = p.user_features, *mf = p.movie_features;

    time_t start,end;
    for (e=0; (e < s.min_epochs) || (rmse <= rmse_last - s.min_improvement); e++) {
        time(&start);
        if (e == s.max_epochs) break;
        cnt++;
        rmse_last = rmse;

        sq = compute_gradient(p, ratings, num_ratings, movie_gradient, 
                user_gradient, s);
        cblas_daxpy(num_m*num_f, -s.lrate/(1-s.lrate*s.K), movie_gradient, 1, mf, 1); 
        cblas_dscal(num_m*num_f, 1-s.lrate*s.K, mf, 1); 
        cblas_daxpy(num_u*num_f, -s.lrate/(1-s.lrate*s.K), user_gradient, 1, uf, 1); 
        cblas_dscal(num_u*num_f, 1-s.lrate*s.K, uf, 1); 
        rmse = sqrt(sq/num_ratings);
        time(&end);
        cout << cnt << " " << rmse << " time: " << difftime(end,start) << "s" << endl;
    }
}

double compute_gradient(Predictor& p, Data *ratings, int num_ratings, 
        double *movie_gradient, double *user_gradient, Settings s) {
    int num_f = p.get_num_features(), 
        num_m = p.get_num_movies(), num_u = p.get_num_users(); 
    for (int i=0; i<(num_m+num_u)*num_f; i++) movie_gradient[i] = 0;
    time_t start,end; time(&start);
    double sq=0;
    #pragma omp parallel
    {
        int user;
        short movie;
        double err, prediction, lcl_sq=0; 
        double *uf, *mf, *temp;
        temp = new double[num_f];
        Data *rating;
        double *lcl_movie_gradient = new double[(num_m+num_u)*num_f] ();
        double *lcl_user_gradient = lcl_movie_gradient + (num_m*num_f);

    // GRADIENT NEEDS TO BE LOCAL, THEN REDUCED
    // NUMB NUTS
    // ALSO, WHY NOT BETTER SPEEDUP?!!!
        #pragma omp for
        for (int i=0; i<num_ratings; i++) {
            rating = ratings + i;
            user = rating->user;
            movie = rating->movie;

            prediction = p.predict(user, movie);
            err = (1.0 * rating->rating - prediction);
            lcl_sq += err*err;

            uf = &p.user_features[user*num_f];
            mf = &p.movie_features[movie*num_f];
            cblas_dcopy(num_f, uf, 1, temp, 1);
            cblas_daxpy(num_f, -err/s.K, mf, 1, temp, 1);
            cblas_daxpy(num_f, s.K, temp, 1, &lcl_user_gradient[user*num_f], 1);
            cblas_dcopy(num_f, mf, 1, temp, 1);
            cblas_daxpy(num_f, -err/s.K, uf, 1, temp, 1);
            cblas_daxpy(num_f, s.K, temp, 1, &lcl_movie_gradient[movie*num_f], 1);
        }
        #pragma omp critical
        {
            sq += lcl_sq;
            cblas_daxpy((num_m+num_u)*num_f, (double)10/num_ratings, lcl_movie_gradient, 1, movie_gradient, 1);
        }
        //for (int i=0; i<(num_m+num_u)*num_f; i++) movie_gradient[i] += lcl_movie_gradient[i]/num_ratings/10;
        delete [] temp;
        delete [] lcl_movie_gradient;
    }
    time(&end); cout << "time: " << difftime(end,start) << "s" << endl;
    return sqrt(sq/num_ratings);
}

void bfgs(Predictor& p, Data *ratings, int num_ratings, Settings s) {
    int num_f=p.get_num_features(), num_m = p.get_num_movies(), num_u = p.get_num_users(); 
    double *movie_gradient = new double[(num_m+num_u)*num_f];
    double *user_gradient = movie_gradient + (num_m*num_f);
    int n = p.get_num_features() * (p.get_num_movies() + p.get_num_users());
    real_1d_array x; x.setcontent(n, p.movie_features);
    double epsg = 0;
    double epsf = 0;
    double epsx = .8;
    real_1d_array scale; scale.setlength(n); for (int i=0; i<n; i++) scale[i] = 1.0;
    ae_int_t maxits = s.max_epochs;
    cout << "maxits: " << maxits << endl;
    cout << "lambda: " << s.K << endl;
    mincgstate state;
    mincgreport rep;
    BFGS_ptr b(p, ratings, num_ratings, movie_gradient, user_gradient, s);

    mincgcreate(n, x, state);
    mincgsetscale(state, scale);
    mincgsetcond(state, epsg, epsf, epsx, maxits);
    mincgsetxrep(state, true);
    cout << "optimizing" << endl;
    alglib::mincgoptimize(state, bfgs_grad, bfgs_callback, &b);
    mincgresults(state, x, rep);

    cout << rep.iterationscount << " iterations " << rep.nfev << " function evaluations" << endl;
    switch(rep.terminationtype) {
        case -2:
            cout << "rounding errors prevent further improvement. X contains best point found." << endl;
            break;
        case -1:
            cout << "incorrect parameters were specified" << endl;
            break;
        case 1:
            cout << "success. relative function improvement is no more than " << epsf << endl;
            break;
        case 2:
            cout << "success. relative step is no more than " << epsx << endl;
            break;
        case 4:
            cout << "success. gradient norm is no more than " << epsg << endl;
            break;
        case 5:
            cout << "MaxIts steps was taken" << endl;
            break;
        case 7:
            cout << "stopping conditions are too stringent, further improvement is impossible" << endl;
            break;
    }
}

void bfgs_grad(const real_1d_array &x, double &f, real_1d_array &grad, void *p) {
    BFGS_ptr *b = (BFGS_ptr *)p;
    int n = x.length();
    for (int i=0; i<n; i++)
        b->predictor.movie_features[i] = x[i];
    f = compute_gradient(b->predictor, b->ratings, b->num_ratings, 
            b->movie_gradient, b->user_gradient, b->settings);
    for (int i=0; i<n; i++)
        grad[i] = b->movie_gradient[i];
}

void bfgs_callback(const real_1d_array &x, double f, void *p) {
    static int i = 0;
    i++;
    cout << "step " << i << " rmse: " << f << endl;
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
