#include "load.h"

using namespace std;

int main() {
    Data *ratings = new Data[100480507];
    // TRY A SMALL ONE BEFORE DOING THIS!!!
    int num_ratings = LoadHistory(ratings, 1);
    DumpBinary(ratings, num_ratings);
    dump_avg(ratings, num_ratings);
    delete [] ratings;
    return 0;
}
