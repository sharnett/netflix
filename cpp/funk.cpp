#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <algorithm>
#include <map>
#include <string>
#include <string.h>
#include "load.h"

using namespace std;

const int MAX_CUSTOMERS = 480190;        // Customers in the entire training set (+1)
const int MAX_MOVIES = 17771;         // Movies in the entire training set (+1)

const int MAX_FEATURES = 5;            // Number of features to use 
double MIN_IMPROVEMENT = 0.01;        // Minimum improvement required to continue current feature
int MIN_EPOCHS = 5;           // Minimum number of epochs per feature
int MAX_EPOCHS = 100;           // Max epochs per feature
const double INIT = 0.1;           // Initialization value for features
const double LRATE = 0.001;         // Learning rate parameter
const double K = 0.015;         // Regularization parameter used to minimize over-fitting

float MovieFeatures[MAX_FEATURES][MAX_MOVIES];
float CustFeatures[MAX_FEATURES][MAX_CUSTOMERS];

typedef unsigned char BYTE;

inline double PredictRating(short movieId, int custId, int feature, float cache, bool bTrailing=true);
inline double PredictRating(short movieId, int custId);
void CalcFeatures(Data *ratings, int num_ratings);
Data *sample(Data *ratings, int sample_size, int num_ratings);
double cost(Data *ratings, int num_ratings);

int main(int argc, char **argv) {
    Data *ratings = new Data[100480507];
    cout << "loading data..." << endl;
    int num_ratings = LoadBinary(ratings); 
    cout << num_ratings << " ratings loaded" << endl;
    for (int f=0; f<MAX_FEATURES; f++) {
        for (int i=0; i<MAX_MOVIES; i++) MovieFeatures[f][i] = (float)INIT;
        for (int i=0; i<MAX_CUSTOMERS; i++) CustFeatures[f][i] = (float)INIT;
    }
//    cout << "trying to shuffle..." << endl;
//    random_shuffle(&ratings[0], &ratings[num_ratings]);
//    for (int i=0; i<10; i++)
//        cout << ratings[i].user << " " << ratings[i].movie << " " << (int)ratings[i].rating << endl;
//    cout << "trying to sample 1000000..." << endl;
    int sample_size = 30200000;
    int cv_size = 200000;
    Data *cv_ratings = sample(ratings, sample_size, num_ratings);
    ratings = cv_ratings + cv_size;
    num_ratings = sample_size - cv_size;
//    for (int i=0; i<10; i++)
//        cout << ratings[i].user << " " << ratings[i].movie << " " << (int)ratings[i].rating << endl;
    CalcFeatures(ratings, num_ratings);
    cout << "training cost: " << cost(ratings, num_ratings) << endl;
    cout << "cross validate cost: " << cost(cv_ratings, cv_size) << endl;
    return 0;
}

void CalcFeatures(Data *ratings, int num_ratings) {
    int f, e, i, custId, cnt = 0;
    Data* rating;
    double err, p, sq, rmse_last, rmse = 2.0;
    short movieId;
    float cf, mf;

    for (f=0; f<MAX_FEATURES; f++) {
        cout << "\n--- Calculating feature: " << f << " ---" << endl;

        // Keep looping until you have passed a minimum number 
        // of epochs or have stopped making significant progress 
        for (e=0; (e < MIN_EPOCHS) || (rmse <= rmse_last - MIN_IMPROVEMENT); e++) {
            cnt++;
            sq = 0;
            rmse_last = rmse;

            for (i=0; i<num_ratings; i++) {
                rating = ratings + i;
                movieId = rating->movie;
                custId = rating->user;

                // Predict rating and calc error
                p = PredictRating(movieId, custId, f, 0, true);
                err = (1.0 * rating->rating - p);
                sq += err*err;
                
                // Cache off old feature values
                cf = CustFeatures[f][custId];
                mf = MovieFeatures[f][movieId];

                // Cross-train the features
                CustFeatures[f][custId] += (float)(LRATE * (err * mf - K * cf));
                MovieFeatures[f][movieId] += (float)(LRATE * (err * cf - K * mf));
            }
            rmse = sqrt(sq/num_ratings);
            cout << cnt << " " << rmse << endl;
        }
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
    double sum = (cache > 0) ? cache : 1; //m_aMovies[movieId].PseudoAvg; 

    // Add contribution of current feature
    sum += MovieFeatures[feature][movieId] * CustFeatures[feature][custId];
    if (sum > 5) sum = 5;
    if (sum < 1) sum = 1;

    // Add up trailing defaults values
    if (bTrailing) {
        sum += (MAX_FEATURES-feature-1) * (INIT * INIT);
        if (sum > 5) sum = 5;
        if (sum < 1) sum = 1;
    }

    return sum;
}

//
// PredictRating
// - This version is used for calculating the final results
// - It loops through the entire list of finished features
//
double PredictRating(short movieId, int custId) {
    double sum = 1; //m_aMovies[movieId].PseudoAvg;

    for (int f=0; f<MAX_FEATURES; f++) 
    {
        sum += MovieFeatures[f][movieId] * CustFeatures[f][custId];
        if (sum > 5) sum = 5;
        if (sum < 1) sum = 1;
    }

    return sum;
}

Data *sample(Data *ratings, int sample_size, int num_ratings) {
    int r;
    for (int i=0; i<sample_size; i++) {
        r = rand()%num_ratings;
        swap(ratings[r], ratings[--num_ratings]);
    }
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
