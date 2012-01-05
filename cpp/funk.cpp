#include <cmath>
#include <algorithm>
#include <boost/random.hpp>
#include <ctime>
#include "load.h"

using namespace std;
using namespace boost;

const int MAX_CUSTOMERS = 480190;        // Customers in the entire training set (+1)
const int MAX_MOVIES = 17771;         // Movies in the entire training set (+1)

int MAX_FEATURES = 25;            // Number of features to use 
double MIN_IMPROVEMENT = 0.001;        // Minimum improvement required to continue current feature
int MIN_EPOCHS = 5;           // Minimum number of epochs per feature
int MAX_EPOCHS = 20;           // Max epochs per feature
const double INIT = 0.1;           // Initialization value for features
const double LRATE = 0.001;         // Learning rate parameter
const double K = 0.015;         // Regularization parameter used to minimize over-fitting

float *MovieFeatures[MAX_MOVIES];
float *CustFeatures[MAX_CUSTOMERS];
float movie_avg[MAX_MOVIES];

inline double PredictRating(short movieId, int custId, int feature, float cache, bool bTrailing=true);
inline double PredictRating(short movieId, int custId);
void CalcFeatures(Data *ratings, int num_ratings);
Data *sample(Data *ratings, int sample_size, int num_ratings);
double cost(Data *ratings, int num_ratings);
void allocate_feature_matrices();
float rndn();

int main(int argc, char **argv) {
    Data *ratings = new Data[100480507];
//    int num_ratings = LoadHistory(ratings);
//    DumpBinary(ratings, num_ratings);
    int num_ratings = LoadBinary(ratings); 
//    int sample_size = 1000000, cv_size = 200000;
    int sample_size = num_ratings, cv_size = 20000000;
    int train_size = sample_size - cv_size;
    Data *cv_ratings = sample(ratings, sample_size, num_ratings);
    Data *train_ratings = cv_ratings + cv_size;

    load_avg(movie_avg);
    allocate_feature_matrices();
    CalcFeatures(train_ratings, train_size);
    cout << "training cost: " << cost(train_ratings, train_size) << endl;
    cout << "cross validate cost: " << cost(cv_ratings, cv_size) << endl;
    return 0;
}

void CalcFeatures(Data *ratings, int num_ratings) {
    int f, e, i, custId, cnt = 0;
    Data* rating;
    double err, p, sq, rmse_last, rmse = 2.0;
    short movieId;
    float cf, mf;

        //cout << "\n--- Calculating feature: " << f << " ---" << endl;

        // Keep looping until you have passed a minimum number 
        // of epochs or have stopped making significant progress 
    for (e=0; (e < MIN_EPOCHS) || (rmse <= rmse_last - MIN_IMPROVEMENT); e++) {
        if (e >= MAX_EPOCHS) break;
        cnt++;
        sq = 0;
        rmse_last = rmse;

        for (i=0; i<num_ratings; i++) {
            rating = ratings + i;
            movieId = rating->movie;
            custId = rating->user;

            // Predict rating and calc error
            p = PredictRating(movieId, custId);
            //p = PredictRating(movieId, custId, f, 0, true);
            err = (1.0 * rating->rating - p);
            sq += err*err;
            
            // Cache off old feature values
            for (f=0; f<MAX_FEATURES; f++) {
                cf = CustFeatures[custId][f];
                mf = MovieFeatures[movieId][f];

                // Cross-train the features
                CustFeatures[custId][f] += (float)(LRATE * (err * mf - K * cf));
                MovieFeatures[movieId][f] += (float)(LRATE * (err * cf - K * mf));
            }
        }
        rmse = sqrt(sq/num_ratings);
        cout << cnt << " " << rmse << endl;
    }
}


//
// PredictRating
// - During training there is no need to loop through all of the features
// - Use a cache for the leading features and do a quick calculation for the trailing
// - The trailing can be optionally removed when calculating a new cache value
//
double PredictRating(short movieId, int custId, int feature, float cache, bool bTrailing) {
    // Get cached value for old features or default to an average
    //double sum = (cache > 0) ? cache : 1; //m_aMovies[movieId].PseudoAvg; 
    double sum = movie_avg[movieId];

    // Add contribution of current feature
    sum += MovieFeatures[movieId][feature] * CustFeatures[custId][feature];
    if (sum > 5) sum = 5;
    if (sum < 1) sum = 1;

    // Add up trailing defaults values
   // if (bTrailing) {
   //     sum += (MAX_FEATURES-feature-1) * (INIT * INIT);
   //     if (sum > 5) sum = 5;
   //     if (sum < 1) sum = 1;
   // }

    return sum;
}

//
// PredictRating
// - This version is used for calculating the final results
// - It loops through the entire list of finished features
//
double PredictRating(short movieId, int custId) {
    //double sum = 1; //m_aMovies[movieId].PseudoAvg;
    double sum = movie_avg[movieId];
    for (int f=0; f<MAX_FEATURES; f++) {
        sum += MovieFeatures[movieId][f] * CustFeatures[custId][f];
        if (sum > 5) sum = 5;
        if (sum < 1) sum = 1;
    }
    return sum;
}

Data *sample(Data *ratings, int sample_size, int num_ratings) {
    int i, r;
    for (i=0; i<sample_size; i++) {
        r = rand()%num_ratings;
        if (r != num_ratings-i-1)
            swap(ratings[r], ratings[num_ratings-i-1]);
    }
    cout << "sampled " << sample_size << " ratings" << endl;
    return &ratings[num_ratings - sample_size];
}

double cost(Data *ratings, int num_ratings) {
    double err, p, sq = 0;
    int user; short movie;
    Data *rating;
    for (int i=0; i<num_ratings; i++) {
        rating = ratings + i;
        movie = rating->movie;
        user = rating->user;

        // Predict rating and calc error
        p = PredictRating(movie, user);
        err = (1.0 * rating->rating - p);
        sq += err*err;
    }
    return sqrt(sq/num_ratings);
}

void allocate_feature_matrices() {
    int i, f;
    for (i=0; i<MAX_MOVIES; i++) {
        MovieFeatures[i] = new float[MAX_FEATURES];
        for (f=0; f<MAX_FEATURES; f++) 
            MovieFeatures[i][f] = rndn();
    }
    for (i=0; i<MAX_CUSTOMERS; i++) {
        CustFeatures[i] = new float[MAX_FEATURES];
        for (f=0; f<MAX_FEATURES; f++) 
            CustFeatures[i][f] = rndn();
    }
    //for (i=0; i<MAX_MOVIES; i++) MovieFeatures[i][f] = (float)INIT;
    //for (i=0; i<MAX_CUSTOMERS; i++) CustFeatures[i][f] = (float)INIT;
}

float rndn() {
    static mt19937 rng(static_cast<unsigned> (time(0)));
    normal_distribution<double> norm_dist(0, .5);
    variate_generator<mt19937&, normal_distribution<double> >  normal_sampler(rng, norm_dist);
    return (float)normal_sampler();
}
