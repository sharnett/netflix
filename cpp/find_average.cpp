#include "globals.h"
#include "load.h"

using namespace std;

extern const int MAX_USERS;   // users in the entire training set 
extern const int MAX_MOVIES;  // movies in the entire training set (+1)

static const int PROBE_SIZE = 1408395;
static const int NON_PROBE_SIZE = 99072112;

int main(int argc, char **argv) {
    Data *ratings, *cv_ratings, *rating;
    double train_avg=0, cv_avg=0, total_avg=0;
    ratings = new Data[NON_PROBE_SIZE]; load_binary(ratings, "cpp/train.bin"); 
    cv_ratings = new Data[PROBE_SIZE]; load_binary(cv_ratings, "cpp/cv.bin"); 

    for (int i=0; i<PROBE_SIZE; i++) {
        rating = cv_ratings + i;
        cv_avg += (int) rating->rating;
    }
    cout << (float) cv_avg/PROBE_SIZE << endl;

    for (int i=0; i<NON_PROBE_SIZE; i++) {
        rating = ratings + i;
        train_avg += (int) rating->rating;
    }
    cout << (float) train_avg/NON_PROBE_SIZE << endl;
    cout << (float) (train_avg+cv_avg) / (PROBE_SIZE+NON_PROBE_SIZE) << endl;

    return 0;
}
