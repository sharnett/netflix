#include <algorithm>
#include "globals.h"
#include "load.h"
#include "predictor.h"
#include "optimizers.h"

using namespace std;

extern const int MAX_USERS;   // users in the entire training set 
extern const int MAX_MOVIES;  // movies in the entire training set (+1)

static bool method = 0; // 0 for sgd, 1 for reg gradient descent
static int sample_size = 1000000;
static const int PROBE_SIZE = 1408395;
static const int NON_PROBE_SIZE = 99072112;

Settings parse_args(int argc, char **argv);
void setup(int& num_ratings, int& num_cv_ratings, Data *& ratings, Data *& cv_ratings);
Data *sample(Data *ratings, int sample_size, int num_ratings);
double cost(Predictor& p, Data *ratings, int num_ratings);

int main(int argc, char **argv) {
    Settings s = parse_args(argc, argv);
    Predictor p(MAX_USERS, MAX_MOVIES, s.num_features);
    int num_ratings, num_cv_ratings;
    Data *ratings, *cv_ratings;

    setup(num_ratings, num_cv_ratings, ratings, cv_ratings);
    cout << "training on " << sample_size << " ratings\n" << endl;

    if (method == 0)
        sgd(p, ratings, sample_size, s);
    else
        gd(p, ratings, sample_size, s);

    cout << "training set cost: " << cost(p, ratings, sample_size) << endl;
    cout << "cross validation cost: " << cost(p, cv_ratings, num_cv_ratings) << endl;

    return 0;
}

void setup(int& num_ratings, int& num_cv_ratings, Data *& ratings, Data *& cv_ratings) {
    ratings = new Data[NON_PROBE_SIZE];
    num_ratings = load_binary(ratings, "cpp/train.bin"); 
    if (sample_size < NON_PROBE_SIZE)
        ratings = sample(ratings, sample_size, num_ratings);
    cv_ratings = new Data[PROBE_SIZE];
    num_cv_ratings = load_binary(cv_ratings, "cpp/cv.bin"); 
}

// essentially this takes sample_size random ratings and shoves them to the back
// of the array, and returns a pointer to the beginning of this shuffled part
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

Settings parse_args(int argc, char **argv) {
    if (argc > 2) {
        cout << "usage: ./funk [-i]" << endl;
        exit(1);
    }
    Settings s;
    if (argc == 2) { // interactive mode
        cout << "enter number of features: ";
        cin >> s.num_features;
        cout << "enter sample size (0 to use all data): ";
        cin >> sample_size;
        if (sample_size <= 0) 
            sample_size = NON_PROBE_SIZE;
        cout << "enter 0 for sgd, 1 for reg grad desc: ";
        cin >> method;
        if (method == 0)
            s.lrate = .001;
        else if (method == 1)
            s.lrate = .0005;
        else {
            cout << "only 0 or 1, silly" << endl;
            exit(1);
        }
        cout << "enter learning rate (0 for ) " << s.lrate << "): ";
        double temp; cin >> temp;
        if (temp > 0)
            s.lrate = temp;
    }
    return s;
}
