#include <algorithm>
#include "globals.h"
#include "load.h"
#include "stdafx.h"
#include "optimization.h"

using namespace std;

extern const int MAX_USERS;   // users in the entire training set 
extern const int MAX_MOVIES;  // movies in the entire training set (+1)

static const int PROBE_SIZE = 1408395;
static const int NON_PROBE_SIZE = 99072112;
static const int MAX_ITS = 0;

static Data *ratings;
static Data *cv_ratings;

const static float avg = 3.6033;
const static float K = .039; // obtained from 'best_averages'
static double *movie_avg;
static double *user_avg;

double cost(Data *ratings, int num_ratings);
double predict(int user, int movie) { return avg + user_avg[user] + movie_avg[movie]; }
void cg(Data *ratings, int num_ratings);
void cg_grad(const alglib::real_1d_array& x, double &f, alglib::real_1d_array& grad, void *p);
double compute_gradient(Data *ratings, int num_ratings, double *movie_gradient, double *user_gradient);
void callback(const alglib::real_1d_array &x, double f, void *p);

struct cg_ptr {
    cg_ptr(Data *r, int n, double *m, double *u):
        ratings(r), num_ratings(n), movie_gradient(m), user_gradient(u) {}
    Data *ratings; 
    int num_ratings;
    double *movie_gradient; 
    double *user_gradient; 
};

// args: min_lambda max_lambda
int main(int argc, char **argv) {
    time_t start, end; 
    double total_time, train_cost, cv_cost;
    ratings = new Data[NON_PROBE_SIZE]; load_binary(ratings, "cpp/train.bin"); 
    cv_ratings = new Data[PROBE_SIZE]; load_binary(cv_ratings, "cpp/cv.bin"); 
    movie_avg = new double[MAX_MOVIES+MAX_USERS] ();
    user_avg = movie_avg + MAX_MOVIES;

    time(&start);
    cg(ratings, NON_PROBE_SIZE);
    time(&end); 
    total_time = difftime(end,start);
    train_cost = cost(ratings, NON_PROBE_SIZE);
    cv_cost = cost(cv_ratings, PROBE_SIZE);

    cout << "total time: " << total_time << "s" << endl;
    cout << "training set cost: " << train_cost << endl;
    cout << "cross validation cost: " << cv_cost << endl;

    dump_averages(movie_avg);
    return 0;
}

double cost(Data *ratings, int num_ratings) {
    double err, prediction, sq = 0;
    int user; short movie;
    Data *rating;
    for (int i=0; i<num_ratings; i++) {
        rating = ratings + i;
        movie = rating->movie;
        user = rating->user;

        prediction = predict(user, movie);
        err = (1.0 * rating->rating - prediction);
        sq += err*err;
    }
    return sqrt(sq/num_ratings);
}

double compute_gradient(Data *ratings, int num_ratings, double *movie_gradient, double *user_gradient) {
    int nm = MAX_MOVIES, nu = MAX_USERS;
    for (int i=0; i<nm+nu; i++) movie_gradient[i] = 0;
    //time_t start,end; time(&start);
    double sq=0;
    #pragma omp parallel
    {
        int user;
        short movie;
        double err, prediction, lcl_sq=0; 
        Data *rating;
        double *lcl_movie_gradient = new double[nm+nu] ();
        double *lcl_user_gradient = lcl_movie_gradient + nm;

        #pragma omp for
        for (int i=0; i<num_ratings; i++) {
            rating = ratings + i;
            user = rating->user;
            movie = rating->movie;

            prediction = predict(user, movie);
            err = (1.0 * rating->rating - prediction);
            lcl_sq += err*err;
            lcl_movie_gradient[movie] += -err + K*movie_avg[movie];
            lcl_user_gradient[user] += -err + K*user_avg[user];
        }
        #pragma omp critical
        {
            sq += lcl_sq;
            cblas_daxpy(nm+nu, (double)10/num_ratings, lcl_movie_gradient, 1, movie_gradient, 1);
        }
        delete [] lcl_movie_gradient;
    }
    //time(&end); cout << "time: " << difftime(end,start) << "s " << sqrt(sq/num_ratings) << endl;
    return sqrt(sq/num_ratings);
}

void cg(Data *ratings, int num_ratings) {
    using namespace alglib;
    int nm = MAX_MOVIES, nu = MAX_USERS, n; n= nm+nu;
    double *movie_gradient = new double[n];
    double *user_gradient = movie_gradient + nm;
    real_1d_array x; x.setlength(n);
    x.setcontent(n, movie_avg);
    double epsg = 0;
    double epsf = 0.001;
    double epsx = 0;
    ae_int_t maxits = MAX_ITS;
    mincgstate state;
    mincgreport rep;
    cg_ptr b(ratings, num_ratings, movie_gradient, user_gradient);

    mincgcreate(n, x, state);
    mincgsetcond(state, epsg, epsf, epsx, maxits);
    mincgsetxrep(state, true);
    cout << "optimizing" << endl;
    alglib::mincgoptimize(state, cg_grad, callback, &b);
    cout << "getting results" << endl;
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

void cg_grad(const alglib::real_1d_array& x, double &f, alglib::real_1d_array& grad, void *p) {
    cg_ptr *b = (cg_ptr *)p;
    int n = x.length();
    for (int i=0; i<n; i++)
        movie_avg[i] = x[i];
    f = compute_gradient(b->ratings, b->num_ratings, b->movie_gradient, b->user_gradient);
    for (int i=0; i<n; i++)
        grad[i] = b->movie_gradient[i];
}

void callback(const alglib::real_1d_array &x, double f, void *p) {
    static int i = 0;
    i++;
    cout << "step " << i << " rmse: " << f << endl;
}
