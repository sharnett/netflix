#include "load.h"
#include "globals.h"

using namespace std;

int main() {
    Data *ratings = new Data[MAX_RATINGS];
    // the '1' tells it to also dump the user dictionary
    int num_ratings = load_history(ratings, 1);
    dump_binary(ratings, num_ratings);
    dump_avg(ratings, num_ratings);
    delete [] ratings;
    return 0;
}
