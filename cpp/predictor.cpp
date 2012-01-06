#include "predictor.h"

Predictor::Predictor(int nu, int nm, int nf) {
    num_users = nu;
    num_movies = nm;
    num_features = nf;
    movie_features = new float*[num_movies];
    user_features = new float*[num_users];
    for (int movie=0; movie<num_movies; movie++) {
        movie_features[movie] = new float[num_features];
        for (int f=0; f<num_features; f++) 
            movie_features[movie][f] = rndn();
    }
    for (int user=0; user<MAX_USERS; user++) {
        user_features[user] = new float[num_features];
        for (int f=0; f<num_features; f++) 
            user_features[user][f] = rndn();
    }
    movie_avg = new float[num_movies];
    load_avg(movie_avg);
}

Predictor::~Predictor() {
    for (int movie=0; movie<num_movies; movie++) 
        delete [] movie_features[movie];
    for (int user=0; user<MAX_USERS; user++)
        delete [] user_features[user];
    delete [] movie_features;
    delete [] user_features;
    delete [] movie_avg;
}

double Predictor::predict(int user, short movie) {
    double sum = movie_avg[movie];
    for (int f=0; f<num_features; f++) 
        sum += movie_features[movie][f] * user_features[user][f];
    if (sum > 5) sum = 5;
    if (sum < 1) sum = 1;
    return sum;
}

float rndn() {
    using namespace boost;
    static mt19937 rng(static_cast<unsigned> (time(0)));
    normal_distribution<float> norm_dist(0, .1);
    variate_generator<mt19937&, normal_distribution<float> >  normal_sampler(rng, norm_dist);
    return normal_sampler();
}
