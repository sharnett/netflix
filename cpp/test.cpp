#include <algorithm>
#include <sqlite3.h>
#include "globals.h"
#include "load.h"
#include "predictor.h"
#include "optimizers.h"
#include <omp.h>
#include <cstdlib>
#include <cstdio>

using namespace std;

extern const int MAX_USERS;   // users in the entire training set 
extern const int MAX_MOVIES;  // movies in the entire training set (+1)

static int method = 0; // 0 for sgd, 1 for reg gradient descent, 2 for conjugate gradient
static bool warm_start = 0; // set to 1 to load features.bin, previous solution
static int sample_size = 1000000;
static const int PROBE_SIZE = 1408395;
static const int NON_PROBE_SIZE = 99072112;

Settings parse_args(int argc, char **argv);
void setup(int& num_ratings, int& num_cv_ratings, Data *& ratings, Data *& cv_ratings);
Data *sample(Data *ratings, int sample_size, int num_ratings);
double cost(Predictor& p, Data *ratings, int num_ratings);
void log(Settings& s, double total_time, double train_cost, double cv_cost);

int comparator(const void *ii, const void *jj) { 
    Data i = *(Data *)ii, j = *(Data *)jj;
    if (i.user > j.user) return 1;
    else if (i.user == j.user && i.movie > j.movie) return 1;
    return -1;
}

int comparator2(const void *ii, const void *jj) { 
    Data i = *(Data *)ii, j = *(Data *)jj;
    if (i.movie > j.movie) return 1;
    else if (i.movie == j.movie && i.user > j.user) return 1;
    return -1;
}

int main(int argc, char **argv) {
    int num_ratings, num_cv_ratings, user, movie;
    Data *ratings, *cv_ratings;
    //sample_size = 500;
    sample_size = NON_PROBE_SIZE;
    setup(num_ratings, num_cv_ratings, ratings, cv_ratings);

    const int nu = 480189;
    const int nm = 17770;
    int *u_rating_ptrs = new int[nu];
    int *m_rating_ptrs = new int[nm+1];

    movie = ratings[0].movie;
    m_rating_ptrs[1] = 0;
    for (int r=1; r<sample_size; r++) {
        if (ratings[r].movie != movie) {
            movie = ratings[r].movie;
            m_rating_ptrs[movie] = r;
        }
    }

    cout << "movie rating counts" << endl;
    Data *start, *end;
    for (int m=1; m<11; m++) {
        start = &ratings[m_rating_ptrs[m]];
        end = &ratings[m_rating_ptrs[m+1]];
        printf("%2d %4lu\n", m, end-start);
    }

    cout << "sorting" << endl;
    time_t t1,t2; time(&t1);
    qsort(ratings, sample_size, sizeof(Data), comparator);
    time(&t2); 
    cout << "total time: " << difftime(t2,t1) << "s" << endl;

    user = ratings[0].user;
    u_rating_ptrs[0] = 0;
    for (int r=1; r<sample_size; r++) {
        if (ratings[r].user != user) {
            user = ratings[r].user;
            u_rating_ptrs[user] = r;
        }
    }

    cout << "user rating counts" << endl;
    for (int u=0; u<10; u++) {
        start = &ratings[u_rating_ptrs[u]];
        end = &ratings[u_rating_ptrs[u+1]];
        printf("%2d %4lu\n", u, end-start);
    }

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

Settings parse_args(int argc, char **argv) {
    Settings s;
    if (argc > 3) {
        cout << "usage: ./funk [-i] [x K]" << endl;
        exit(1);
    }
    if (argc == 3) {
        s.num_features = 50;
        sample_size = NON_PROBE_SIZE;
        method = 2;
        s.max_epochs = 0;
        s.K = atof(argv[2]);
        cout << "using K = " << s.K << endl;
    }
    if (argc == 2) { // interactive mode
        cout << "enter number of features: ";
        cin >> s.num_features;
        cout << "enter sample size (0 to use all data): ";
        cin >> sample_size;
        if (sample_size <= 0) 
            sample_size = NON_PROBE_SIZE;
        cout << "enter 0 for sgd, 1 for reg grad desc, 2 for bfgs: ";
        cin >> method;
        if (method == 0)
            s.lrate = .001;
        else if (method == 1)
            s.lrate = .0005;
        else if (method != 2) {
            cout << "only 0 or 1 or 2, silly" << endl;
            exit(1);
        }
        if (method == 0 || method == 1) {
            cout << "enter learning rate (0 for " << s.lrate << "): ";
            double temp; cin >> temp;
            if (temp > 0)
                s.lrate = temp;
        }
        cout << "max iterations: ";
        cin >> s.max_epochs;
        cout << "regularizataion parameter (.015): ";
        cin >> s.K;
        cout << "minimum improvement (.001): ";
        cin >> s.min_improvement;
    }
    return s;
}

void log(Settings& s, double total_time, double train_cost, double cv_cost) {
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
    if (method == 0) method_string = "sgd";
    else if (method == 1) method_string = "gd";
    else if (method == 2) {
        method_string = "cg";
        s.lrate = 0;
    }
    sprintf(temp, "'%s', %d, %f, %f, %f, %f, %f, %d, %d)",
            method_string.c_str(), s.num_features, cv_cost, total_time, train_cost, s.lrate,
            s.K, sample_size, warm_start);
    sql += temp;

    rc = sqlite3_exec(db, sql.c_str(), NULL, 0, &zErrMsg);
    if (rc!=SQLITE_OK) {
        fprintf(stderr, "SQL error: %s\n", zErrMsg);
        sqlite3_free(zErrMsg);
    }
    sqlite3_close(db);

}
