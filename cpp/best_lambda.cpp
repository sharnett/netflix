#include <algorithm>
#include <sqlite3.h>
#include "globals.h"
#include "load.h"
#include "predictor.h"
#include "optimizers.h"
#include "fmin.h"

using namespace std;

extern const int MAX_USERS;   // users in the entire training set 
extern const int MAX_MOVIES;  // movies in the entire training set (+1)

static const int PROBE_SIZE = 1408395;
static const int NON_PROBE_SIZE = 99072112;

static Data *ratings;
static Data *cv_ratings;
static Settings s;
static int iteration = 0;

double cost(Predictor& p, Data *ratings, int num_ratings);
void log(Settings& s, double total_time, double train_cost, double cv_cost);
double f(double K);

// args: num_features min_lambda max_lambda
int main(int argc, char **argv) {
    double min_lambda, max_lambda;
    if (argc != 4) {
        cout << "usage: ./best_lambda num_features min_lambda max_lambda" << endl;
        exit(1);
    }

    s.max_epochs = 0;
    s.num_features = atoi(argv[1]);
    min_lambda = atof(argv[2]);
    max_lambda = atof(argv[3]);
    ratings = new Data[NON_PROBE_SIZE]; load_binary(ratings, "cpp/train.bin"); 
    cv_ratings = new Data[PROBE_SIZE]; load_binary(cv_ratings, "cpp/cv.bin"); 

    double best_lambda = fminbnd(f, min_lambda, max_lambda);
    cout << "optimal lambda is " << best_lambda << endl;

    return 0;
}

double f(double K) {
    s.K = K;
    cout << "iteration " << ++iteration;
    cout << " using K = " << s.K << endl;
    time_t start,end; time(&start);
    Predictor p(MAX_USERS, MAX_MOVIES, s.num_features);
    bfgs(p, ratings, NON_PROBE_SIZE, s);
    time(&end); 
    
    double total_time, train_cost, cv_cost;
    total_time = difftime(end,start);
    train_cost = cost(p, ratings, NON_PROBE_SIZE);
    cv_cost = cost(p, cv_ratings, PROBE_SIZE);

    cout << "total time: " << total_time << "s" << endl;
    cout << "training set cost: " << train_cost << endl;
    cout << "cross validation cost: " << cv_cost << endl;

    log(s, total_time, train_cost, cv_cost);
    cout << endl;
    return cv_cost;
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

void log(Settings& s, double total_time, double train_cost, double cv_cost) {
    if (NON_PROBE_SIZE < 90000000) return;
    sqlite3 *db;
    char *zErrMsg = 0;
    int rc;
    string sql;
    rc = sqlite3_open_v2("log.db", &db, SQLITE_OPEN_READWRITE, NULL);
    if (rc) {
        fprintf(stderr, "Can't open database: %s\n", sqlite3_errmsg(db));
        sqlite3_close(db);
        exit(1);
    }

    sql = "INSERT INTO log "
          "(datetime, method, num_features, cv_cost, time, train_cost, learning_rate, "
          "regularizer, sample_size, warm_start) "
          "VALUES (datetime('now'), ";
    char temp[100];
    string method_string;
    method_string = "cg";
    s.lrate = 0;
    sprintf(temp, "'%s', %d, %f, %f, %f, %f, %f, %d, %d)",
            method_string.c_str(), s.num_features, cv_cost, total_time, train_cost, s.lrate,
            s.K, NON_PROBE_SIZE, 0);
    sql += temp;

    rc = sqlite3_exec(db, sql.c_str(), NULL, 0, &zErrMsg);
    if (rc!=SQLITE_OK) {
        fprintf(stderr, "SQL error: %s\n", zErrMsg);
        sqlite3_free(zErrMsg);
    }
    sqlite3_close(db);

}
