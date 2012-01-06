#ifndef globals_H
#define globals_H

typedef unsigned char BYTE;

struct Data {
    int user;
    short movie;
    BYTE rating;
};

const int MAX_USERS = 480190;      // users in the entire training set (+1)
const int MAX_MOVIES = 17771;      // movies in the entire training set (+1)
const int MAX_RATINGS = 100480507; // total number of ratings

#endif
