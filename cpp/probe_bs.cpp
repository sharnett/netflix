// probe_bs takes all the ratings from probe.txt and removes them from the original
// data, leaving behind pure training data. the removed ratings are used as cross-
// validation data
//
// the way I did this ended up being a larger PITA than expected

#include <list>
#include <algorithm>
#include "load.h"

using namespace std;

int get_probe_movies(list<int> probe_movies[]);
void remap_probe_movies(list<int> probe_movies[], map<int, int> user_dict);
void split_train_and_cv_data(list<int> probe_movies[], int num_cv_ratings);

int main() {
    list<int> probe_movies[MAX_MOVIES]; 
    int num_cv_ratings = get_probe_movies(probe_movies);
    map<int, int> user_dict; load_user_dict(user_dict);
    remap_probe_movies(probe_movies, user_dict);
    split_train_and_cv_data(probe_movies, num_cv_ratings);
    return 0;
}

// an array of lists, one list per movie, each list contains all the users
// in probe.txt who rated that movie
int get_probe_movies(list<int> probe_movies[]) {
    string probe_file = get_data_folder() + "probe.txt";
    cout << "trying to open " << probe_file << endl;
    FILE *stream;
    if ((stream = fopen(probe_file.c_str(), "r")) == NULL) {
        cout << "error opening " << probe_file << endl;
        exit(1);
    }
    int user, movie=0, i=0; 
    char buf[1000];
    char *temp;
    while(fgets(buf, 1000, stream)) {
        temp = strchr(buf, ':');
        if (temp) 
            movie = atoi(buf);
        else {
            user = atoi(buf);
            probe_movies[movie].push_back(user);
            i++;
        }
    }
    cout << i << " probe ratings loaded" << endl;
    fclose(stream);
    return i;
}

// convert the user IDs in the array of lists above to compact form
void remap_probe_movies(list<int> probe_movies[], map<int, int> user_dict) {
    cout << "converting probe movie IDs to compact form..." << endl;
    list<int>::iterator it;
    list<int> *ml;
    for (int movie=1; movie<MAX_MOVIES; movie++) {
        ml = &probe_movies[movie];
        for (it = ml->begin(); it != ml->end(); ++it)
            *it = user_dict[*it];
    }
}

// go through the main data structure and split off the ratings that were
// in probe.txt to create two sets of data
void split_train_and_cv_data(list<int> probe_movies[], int num_cv_ratings) {
    int num_ratings, num_train_ratings;
    Data *ratings, *train_ratings, *cv_ratings;
    ratings = new Data[MAX_RATINGS];
    num_ratings = load_binary(ratings); 
    num_train_ratings = num_ratings - num_cv_ratings;
    cv_ratings = new Data[num_cv_ratings];
    train_ratings = new Data[num_train_ratings];

    Data *orig = ratings, *cv = cv_ratings, *train = train_ratings;
    list<int>::iterator it;
    list<int> *ml;
    int i=0, j=0, k=0;
    cout << "trying to split original data into training and cross-validation"
       " sets..." << endl;
    time_t start,end; time(&start);
    // iterate through list of probe movies
    for (int probe_movie=1; probe_movie<MAX_MOVIES; probe_movie++) {
        // find the probe_movie in the original data
        // note that original data is sorted by movie
        while (orig->movie != probe_movie)
            orig = ratings + ++i; 
        // ml is the list of users for this probe movie
        ml = &probe_movies[probe_movie];
        // for a particular probe movie, loop through all users in the 
        // original data. 
        while (orig->movie == probe_movie && i<MAX_RATINGS) {
            if (i%1000000 == 0)
                cout << "doing rating " << i << endl;
            // try to find the user in the probe user list for this movie
            it = find(ml->begin(), ml->end(), orig->user);
            // if it's in the list, remove it, and add the data point to the 
            // cross-validation set 
            if (it != ml->end()) {
                ml->erase(it);
                cv->user = orig->user;
                cv->movie = orig->movie;
                cv++->rating = orig->rating;
                j++;
            }
            // else add the data point to the training set
            else {
                train->user = orig->user;
                train->movie = orig->movie;
                train++->rating = orig->rating;
                k++;
            }
            orig = ratings + ++i;
        }
    }
    time(&end);
    cout << "time: " << difftime(end,start) << "s" << endl;
    delete [] ratings;

    // dump it all
    dump_binary(train_ratings, num_train_ratings, "cpp/train.bin");
    dump_binary(cv_ratings, num_cv_ratings, "cpp/cv.bin");
    delete [] train_ratings;
    delete [] cv_ratings;
}
