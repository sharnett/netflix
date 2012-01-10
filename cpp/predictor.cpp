#include "predictor.h"

Predictor::Predictor(int nu, int nm, int nf) {
    num_users = nu;
    num_movies = nm;
    num_features = nf;
    movie_features = new float[num_movies*num_features];
    user_features = new float[num_users*num_features];
    for (int movie=0; movie<num_movies; movie++) {
        for (int f=0; f<num_features; f++) 
            movie_features[movie*num_features + f] = rndn();
    }
    for (int user=0; user<num_users; user++) {
        for (int f=0; f<num_features; f++) 
            user_features[user*num_features + f] = rndn();
    }
    movie_avg = new float[num_movies];
    load_avg(movie_avg);
}

Predictor::~Predictor() {
    delete [] movie_features;
    delete [] user_features;
    delete [] movie_avg;
}

double Predictor::predict(int user, short movie) {
    double sum = movie_avg[movie];
//  BLAS is about 10% faster than naive dot product for me
    sum += cblas_sdot(num_features, &movie_features[movie*num_features],
            1, &user_features[user*num_features], 1);
//    for (int f=0; f<num_features; f++) 
//        sum += movie_features[movie][f] * user_features[user][f];
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
