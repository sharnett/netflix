#include "load.h"

using namespace std;

int main() {
    Data *ratings = new Data[100480507];
    int num_ratings = LoadHistory(ratings);
    DumpBinary(ratings, num_ratings);
    dump_avg(ratings, num_ratings);
    return 0;
}
