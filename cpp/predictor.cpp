#include "predictor.h"

Predictor::Predictor(int nu, int nm, int nf) {
    srand(0);
    //srand(time(0));
    num_users = nu; num_movies = nm; num_features = nf;
    movie_features = new float[(nm+nu)*nf];
    user_features = movie_features + (nm*nf);
    for (int i=0; i<(nm+nu)*nf; i++) 
        movie_features[i] = rndn();
    movie_avg = new double[nu+nm];
    load_averages(movie_avg);
    user_avg = movie_avg + nm;
    average = 3.6033;
}

Predictor::~Predictor() {
    delete [] movie_features;
//    delete [] user_features;
    delete [] movie_avg;
}

float Predictor::predict(int user, short movie) {
    float sum = average + movie_avg[movie] + user_avg[user];
    sum += cblas_sdot(num_features, &movie_features[movie*num_features],
            1, &user_features[user*num_features], 1);
//    for (int f=0; f<num_features; f++) 
//        sum += movie_features[movie][f] * user_features[user][f];
    if (sum > 5) sum = 5;
    if (sum < 1) sum = 1;
    return sum;
}

double rndn() {
    return 3*((double)rand()/RAND_MAX/5 - .1);
    // the below uses the boost library to get normal random numbers
    // not really necessary
/*    using namespace boost;
    static mt19937 rng(static_cast<unsigned> (time(0)));
    normal_distribution<double> norm_dist(0, .1);
    variate_generator<mt19937&, normal_distribution<double> >  
            normal_sampler(rng, norm_dist);
    return normal_sampler(); */
}
