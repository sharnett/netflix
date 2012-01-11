#ifndef globals_H
#define globals_H

#include <iostream>
#include <ctime>

typedef unsigned char BYTE;

struct Data {
    int user;
    short movie;
    BYTE rating;
};

const int MAX_USERS = 480189;      // users in the entire training set
// the movie IDs start at 1, so just added a dummy movie 0 for convenience
const int MAX_MOVIES = 17771;      // movies in the entire training set (+1)
const int MAX_RATINGS = 100480507; // total number of ratings

#endif
