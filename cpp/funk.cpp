#include <algorithm>
#include <ctime>
#include "load.h"
#include "predictor.h"
#include "optimizers.h"

using namespace std;
using namespace boost;

extern double MIN_IMPROVEMENT;        // Minimum improvement required to continue current feature
extern int MAX_EPOCHS;           // Max epochs per feature
extern double LRATE;         // Learning rate parameter

extern const int MAX_USERS;        // users in the entire training set (+1)
extern const int MAX_MOVIES;         // movies in the entire training set (+1)
extern const int MAX_RATINGS; 

static int num_features = 5;            // Number of features to use 
static int sample_size = 1000000;
static int cv_size = 200000;
static bool method = 0; // 0 for sgd, 1 for reg gd

Data *sample(Data *ratings, int sample_size, int num_ratings);
double cost(Predictor& p, Data *ratings, int num_ratings);
void parse_args(int argc, char **argv);

int main(int argc, char **argv) {
    parse_args(argc, argv);
    Data *ratings, *train_ratings, *cv_ratings;
    ratings = new Data[MAX_RATINGS];
    int num_ratings = LoadBinary(ratings); 
    int train_size = sample_size - cv_size;
    cout << "using " << sample_size << " ratings: " << train_size << " training, " <<
        cv_size << " cross validation" << endl;
    cv_ratings = sample(ratings, sample_size, num_ratings);
    train_ratings = cv_ratings + cv_size;
    Predictor p(MAX_USERS, MAX_MOVIES, num_features);

    if (method == 0)
        sgd(p, train_ratings, train_size);
    else
        gd(p, train_ratings, train_size);
    cout << "training set cost: " << cost(p, train_ratings, train_size) << endl;
    cout << "cross validation set cost: " << cost(p, cv_ratings, cv_size) << endl;

    delete [] ratings;
    return 0;
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

double cost(Predictor& p, Data *ratings, int num_ratings) {
    double err, prediction, sq = 0;
    int user; short movie;
    Data *rating;
    for (int i=0; i<num_ratings; i++) {
        rating = ratings + i;
        movie = rating->movie;
        user = rating->user;

        prediction = p.predict(user, movie);
        err = (1.0 * rating->rating - prediction);
        sq += err*err;
    }
    return sqrt(sq/num_ratings);
}

void parse_args(int argc, char **argv) {
    if (argc > 2) {
        cout << "usage: ./funk [-i]" << endl;
        exit(1);
    }
    set_defaults();
    if (argc == 2) { // interactive mode
        cout << "enter number of features: ";
        cin >> num_features;
        cout << "enter sample size (0 to use all data): ";
        cin >> sample_size;
        if (sample_size <= 0) 
            sample_size = MAX_RATINGS;
        cv_size = sample_size/10;
        cout << "enter 0 for sgd, 1 for reg grad desc: ";
        cin >> method;
        cout << "enter learning rate (0 for .001): ";
        cin >> LRATE;
        if (LRATE <= 0)
            LRATE = .001;
    }
}
