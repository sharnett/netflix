// THIS DOESN'T WORK

#include "optimizers.h"

using namespace std;

static time_t start, end;

// maybe have ALS in a separate file since it is such a beast
void als(Predictor& p, Data *ratings, Data *ratings2, int num_ratings, Settings s) {
    // need one pass through the data to get
    //    - number of ratings for each user
    //    - number of ratings for each movie
    // or perhaps better, a pointer to the first rating for each user/movie
    // note that this info is built into the slick data representation
    //
// repeat until error is small
//   solve for U with M fixed
//   solve for M with U fixed
//   calculate error
    cout << "doing alternating least squares" << endl;
    int e, cnt = 0, nf=p.get_num_features(),
        nm = p.get_num_movies(), nu = p.get_num_users(), user, movie;
    float rmse_last=3, rmse = 2.0;
    float *uf = p.user_features, *mf = p.movie_features;

    int *u_rating_ptrs, *m_rating_ptrs;
    u_rating_ptrs = new int[nu]; m_rating_ptrs = new int[nm];
    //for (int r=0; r<num_ratings; r++) {
        //movie = ratings[i].movie;
    //}

    for (e=0; (e < s.min_epochs) || (rmse <= rmse_last - s.min_improvement); e++) {
        time(&start);
        if (e == s.max_epochs) break;
        cnt++;
        rmse_last = rmse;
        solveU(p, ratings, num_ratings, s.K);
        solveM(p, ratings2, num_ratings, s.K);
        rmse = cost(p, ratings, num_ratings);
        time(&end);
        cout << cnt << " " << rmse << " time: " << difftime(end,start) << "s" << endl;
    }
}

void solveU(Predictor& p, Data *ratings, int num_ratings, double K) {
    int nf=p.get_num_features(), nm = p.get_num_movies(), nu = p.get_num_users();
    // global variables
    #pragma omp parallel
    {
        // local variables
        #pragma omp for
        for (int u=0; u<nu; u++) {
            int nr = u; //.num_ratings;
            // build temporary matrices
            // note: building R not necessary, just just use pointer
            // some loop of this kind is necessary to build M though
            for (int r=0; r<nr; r++) {
                //R[r] = ratings[r];
                //M[r] = movie_features[r];
            }
            //U[i] = (M M^T + K nr I)^-1 M R^T
        }
        #pragma omp critical
        {
            // all gather?
            // i dont think this is necessary, the parfor loop does it all
        }
        // clean up local variables
    }
    // clean up global variables
}

void solveM(Predictor& p, Data *ratings, int num_ratings, double K) {
    int nf=p.get_num_features(), nm = p.get_num_movies(), nu = p.get_num_users();
    // global variables
    #pragma omp parallel
    {
        // local variables
        #pragma omp for
        for (int m=0; m<nm; m++) {
            // for loop
        }
        #pragma omp critical
        {
            // all gather
        }
        // clean up local variables
    }
    // clean up global variables
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
