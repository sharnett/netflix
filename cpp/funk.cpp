#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <map>
#include <string>
#include <string.h>
#include <vector>
#include <algorithm>
using namespace std;

//===================================================================
//
// Constants and Type Declarations
//
//===================================================================
//const char *TRAINING_PATH = "/Users/srharnett/Downloads/download/training_set/";
//const char *TRAINING_FILE = "/Users/srharnett/Downloads/download/training_set/%s";
const char *TRAINING_PATH = "/home/sean/Documents/download/training_set/";
const char *TRAINING_FILE = "/home/sean/Documents/download/training_set/%s";
const char *TEST_PATH = "/home/sean/Documents/download/%s";
const char *PREDICTION_FILE = "/home/sean/Documents/download/prediction.txt";

#define MAX_RATINGS     100480508     // Ratings in entire training set (+1)
#define MAX_CUSTOMERS   480190        // Customers in the entire training set (+1)
#define MAX_MOVIES      17771         // Movies in the entire training set (+1)
#define MAX_FEATURES    2            // Number of features to use 
#define MIN_EPOCHS      5           // Minimum number of epochs per feature
#define MAX_EPOCHS      15           // Max epochs per feature
//#define MAX_FEATURES    64            // Number of features to use 
//#define MIN_EPOCHS      120           // Minimum number of epochs per feature
//#define MAX_EPOCHS      200           // Max epochs per feature

#define MIN_IMPROVEMENT 0.0001        // Minimum improvement required to continue current feature
#define INIT            0.1           // Initialization value for features
#define LRATE           0.001         // Learning rate parameter
#define K               0.015         // Regularization parameter used to minimize over-fitting

typedef unsigned char BYTE;
typedef map<int, int> IdMap;
typedef IdMap::iterator IdItr;

struct Movie
{
    int         RatingCount;
    int         RatingSum;
    double      RatingAvg;            
    double      PseudoAvg;            // Weighted average used to deal with small movie counts 
};

struct MovieRating {
    short movie;
    BYTE rating;
};

bool sort_by_rating(const MovieRating& x, const MovieRating& y) {
    return (x.rating <= y.rating);
}

struct Customer
{
    int         CustomerId;
    int         RatingCount;
    int         RatingSum;
};

struct Data
{
    int         CustId;
    short       MovieId;
    BYTE        Rating;
    float       Cache;
};

struct Record {
    short counts[5];
    short *movies;
};


class Engine 
{
private:
    int             m_nRatingCount;                                 // Current number of loaded ratings
    Data            m_aRatings[MAX_RATINGS];                        // Array of ratings data
    Record          records[MAX_CUSTOMERS];                    // Array of customer metrics
    short           special_movies[MAX_RATINGS];
    Movie           m_aMovies[MAX_MOVIES];                          // Array of movie metrics
    Customer        m_aCustomers[MAX_CUSTOMERS];                    // Array of customer metrics
    float           m_aMovieFeatures[MAX_FEATURES][MAX_MOVIES];     // Array of features by movie (using floats to save space)
    float           m_aCustFeatures[MAX_FEATURES][MAX_CUSTOMERS];   // Array of features by customer (using floats to save space)
    IdMap           m_mCustIds;                                     // Map for one time translation of ids to compact array index

    inline double   PredictRating(short movieId, int custId, int feature, float cache, bool bTrailing=true);
    inline double   PredictRating(short movieId, int custId);

public:
    Engine(void);
    ~Engine(void) { };

    void            CalcMetrics();
    void            CalcFeatures();
    void            LoadHistory();
    void            LoadHistory2();
    void            DumpHistory2();
    void            DumpBinary();
    void            DumpBinary2();
    void            LoadBinary();
    void            LoadBinary2();
    void            ProcessTest(char *pwzFile);
    void            ProcessFile(char *pwzFile);
};


//===================================================================
//
// Program Main
//
//===================================================================
int main(int argc, char **argv)
{
    Engine* engine = new Engine();

//    engine->LoadHistory();
//    engine->CalcMetrics();
//    engine->DumpBinary();
//    engine->LoadBinary();
//    engine->LoadHistory2();
//    engine->DumpBinary2();
    engine->LoadBinary2();
    //engine->DumpHistory2();
    engine->CalcFeatures();
    engine->ProcessTest("qualifying.txt");

    cout << "Done" << endl;
    getchar();
    
    return 0;
}


//===================================================================
//
// Engine Class 
//
//===================================================================

//-------------------------------------------------------------------
// Initialization
//-------------------------------------------------------------------

Engine::Engine(void)
{
    m_nRatingCount = 0;

    for (int f=0; f<MAX_FEATURES; f++)
    {
        for (int i=0; i<MAX_MOVIES; i++) m_aMovieFeatures[f][i] = (float)INIT;
        for (int i=0; i<MAX_CUSTOMERS; i++) m_aCustFeatures[f][i] = (float)INIT;
    }
}

//-------------------------------------------------------------------
// Calculations - This Paragraph contains all of the relevant code
//-------------------------------------------------------------------

//
// CalcMetrics
// - Loop through the history and pre-calculate metrics used in the training 
// - Also re-number the customer id's to fit in a fixed array
//
void Engine::CalcMetrics()
{
    int i, cid;
    IdItr itr;

    printf("\nCalculating intermediate metrics\n");

    // Process each row in the training set
    for (i=0; i<m_nRatingCount; i++)
    {
        Data* rating = m_aRatings + i;

        // Increment movie stats
        //m_aMovies[rating->MovieId].RatingCount++;
        //m_aMovies[rating->MovieId].RatingSum += rating->Rating;
        
        // Add customers (using a map to re-number id's to array indexes) 
        itr = m_mCustIds.find(rating->CustId); 
        if (itr == m_mCustIds.end())
        {
            cid = 1 + (int)m_mCustIds.size();

            // Reserve new id and add lookup
            m_mCustIds[rating->CustId] = cid;

            // Store off old sparse id for later
            m_aCustomers[cid].CustomerId = rating->CustId;

            // Init vars to zero
         //   m_aCustomers[cid].RatingCount = 0;
         //   m_aCustomers[cid].RatingSum = 0;
        }
        else
        {
            cid = itr->second;
        }

        // Swap sparse id for compact one
        rating->CustId = cid;

        //m_aCustomers[cid].RatingCount++;
        //m_aCustomers[cid].RatingSum += rating->Rating;
    }

    // Do a follow-up loop to calc movie averages
    //for (i=0; i<MAX_MOVIES; i++)
    //{
    //    Movie* movie = m_aMovies+i;
    //    movie->RatingAvg = movie->RatingSum / (1.0 * movie->RatingCount);
    //    movie->PseudoAvg = (3.23 * 25 + movie->RatingSum) / (25.0 + movie->RatingCount);
    //}
}

//
// CalcFeatures
// - Iteratively train each feature on the entire data set
// - Once sufficient progress has been made, move on
//
void Engine::CalcFeatures()
{
    int f, e, i, cnt = 0;
    Data* rating;
    double err, p, sq, rmse_last, rmse = 2.0;
    short movie;
    float cf, mf;

    short *counts;
    short *movies;

    for (f=0; f<MAX_FEATURES; f++)
    {
        printf("\n--- Calculating feature: %d ---\n", f);

        // Keep looping until you have passed a minimum number 
        // of epochs or have stopped making significant progress 
        for (e=0; (e < MIN_EPOCHS) || (rmse <= rmse_last - MIN_IMPROVEMENT); e++)
        {
            cnt++;
            sq = 0;
            rmse_last = rmse;

            time_t start,end;
            double dif;
            time(&start);

            for (int user=1; user<MAX_CUSTOMERS; user++) {
                counts = records[user].counts;
                movies = records[user].movies;
                int k = 0;
                for (int i=0; i<5; i++) {
                    for (int j=0; j<counts[i]; j++) {
                        movie = movies[j+k];
                        p = PredictRating(movie, user, f, 0, true);
                        err = (1.0*(i+1) - p);
                        sq += err*err;
                        cf = m_aCustFeatures[f][user];
                        mf = m_aMovieFeatures[f][movie];
                        m_aCustFeatures[f][user] += (float)(LRATE * (err * mf - K * cf));
                        m_aMovieFeatures[f][movie] += (float)(LRATE * (err * cf - K * mf));
                    }
                }
            }
/*            for (i=0; i<m_nRatingCount; i++) {
                rating = m_aRatings + i;
                movieId = rating->MovieId;
                custId = rating->CustId;

                // Predict rating and calc error
                p = PredictRating(movieId, custId, f, rating->Cache, true);
                err = (1.0 * rating->Rating - p);
                sq += err*err;
                
                // Cache off old feature values
                cf = m_aCustFeatures[f][custId];
                mf = m_aMovieFeatures[f][movieId];

                // Cross-train the features
                m_aCustFeatures[f][custId] += (float)(LRATE * (err * mf - K * cf));
                m_aMovieFeatures[f][movieId] += (float)(LRATE * (err * cf - K * mf));
            }
*/
            
            rmse = sqrt(sq/(MAX_RATINGS -1));
            //rmse = sqrt(sq/m_nRatingCount);
                  
            time(&end);
            dif = difftime (end,start);
            cout << "sweep: " << cnt << " rmse: " << rmse << " time: " << dif << endl;
            //printf("     <set x='%d' y='%f' />\n",cnt,rmse);
        }

        sq = 0;
        for (int user=1; user<MAX_CUSTOMERS; user++) {
            counts = records[user].counts;
            movies = records[user].movies;
            int k = 0;
            for (int i=0; i<5; i++) {
                for (int j=0; j<counts[i]; j++) {
                    movie = movies[j+k];
                    p = PredictRating(movie, user, f, 0, true);
                    err = (1.0*(i+1) - p);
                    sq += err*err;
                    cf = m_aCustFeatures[f][user];
                    mf = m_aMovieFeatures[f][movie];
                    m_aCustFeatures[f][user] += (float)(LRATE * (err * mf - K * cf));
                    m_aMovieFeatures[f][movie] += (float)(LRATE * (err * cf - K * mf));
                }
            }
        }
        rmse = sqrt(sq/(MAX_RATINGS -1));
        cout << "final score: " << rmse << endl;

        // Cache off old predictions
        //for (i=0; i<m_nRatingCount; i++)
        //{
        //    rating = m_aRatings + i;
        //    rating->Cache = (float)PredictRating(rating->MovieId, rating->CustId, f, rating->Cache, false);
        //}            
    }
}

//
// PredictRating
// - During training there is no need to loop through all of the features
// - Use a cache for the leading features and do a quick calculation for the trailing
// - The trailing can be optionally removed when calculating a new cache value
//
double Engine::PredictRating(short movieId, int custId, int feature, float cache, bool bTrailing)
{
    // Get cached value for old features or default to an average
    double sum = (cache > 0) ? cache : 1; //m_aMovies[movieId].PseudoAvg; 

    // Add contribution of current feature
    sum += m_aMovieFeatures[feature][movieId] * m_aCustFeatures[feature][custId];
    if (sum > 5) sum = 5;
    if (sum < 1) sum = 1;

    // Add up trailing defaults values
    if (bTrailing)
    {
        sum += (MAX_FEATURES-feature-1) * (INIT * INIT);
        if (sum > 5) sum = 5;
        if (sum < 1) sum = 1;
    }

    return sum;
}

//
// PredictRating
// - This version is used for calculating the final results
// - It loops through the entire list of finished features
//
double Engine::PredictRating(short movieId, int custId)
{
    double sum = 1; //m_aMovies[movieId].PseudoAvg;

    for (int f=0; f<MAX_FEATURES; f++) 
    {
        sum += m_aMovieFeatures[f][movieId] * m_aCustFeatures[f][custId];
        if (sum > 5) sum = 5;
        if (sum < 1) sum = 1;
    }

    return sum;
}

//-------------------------------------------------------------------
// Data Loading / Saving
//-------------------------------------------------------------------

//
// LoadHistory
// - Loop through all of the files in the training directory
//
void Engine::LoadHistory() {
//    for (int i = 1; i < 100; i++) {
    for (int i = 1; i < MAX_MOVIES; i++) {
        char data_file[100];
        sprintf(data_file, "%smv_00%05d.txt", TRAINING_PATH, i);
        this->ProcessFile(data_file);
    }
}

void Engine::DumpBinary() {
    FILE* f = fopen("binary.txt", "w");
    fwrite(m_aRatings, sizeof(Data), m_nRatingCount, f);
    fclose(f);
}

void Engine::DumpBinary2() {
    FILE* f = fopen("special1.bin", "w");
    fwrite(records, sizeof(Record), MAX_CUSTOMERS, f);
    fclose(f);
    f = fopen("special2.bin", "w");
    fwrite(special_movies, sizeof(short), MAX_RATINGS-1, f);
    fclose(f);
}

void Engine::LoadBinary2() {
    int user = 0;
    FILE* f = fopen("special1.bin", "r");
    while(fread(&records[user], sizeof(Record), 1, f))
        user++;
    fclose(f);
    int n = 0;
    f = fopen("special2.bin", "r");
    while (fread(&special_movies[n], sizeof(short), 1, f))
        n++;
    fclose(f);

    ptrdiff_t d = special_movies - records[1].movies;
    for (int i=1; i<user; i++)
        records[i].movies += d;
}

void Engine::LoadBinary() {
    FILE* f = fopen("binary.txt", "r");
    // should be able to do this all in one shot, no loop
    // couldnt figure it out
    while(fread(&m_aRatings[m_nRatingCount], sizeof(Data), 1, f) == 1)
        m_nRatingCount++;
    fclose(f);
}

void Engine::DumpHistory2() {
    FILE *f = fopen("special.txt", "w");
    short *counts;
    int total;
    for (int user = 1; user < MAX_CUSTOMERS; user++) {
        counts = records[user].counts;
        fprintf(f, "%d %d %d %d %d %d ", user, counts[0], counts[1], counts[2], counts[3], counts[4]);
        total = 0;
        for (int i=0; i<5; i++) 
            total += counts[i];
        for (int i=0; i<total; i++)
            fprintf(f, "%d ", records[user].movies[i]);
        fprintf(f, "\n");
    }
    fclose(f);
}

void Engine::LoadHistory2() {
    cout << "making silly new data structure" << endl;
    Data* rating;
    BYTE value;
    int user;
    short movie;
    // determine how much space the movies list needs for each record
    for (int i=0; i<m_nRatingCount; i++) {
        rating = m_aRatings + i;
        user = rating->CustId;
        value = rating->Rating;
        records[user].counts[value-1]++;
    }

    int total = 0;
    records[1].movies = special_movies;
    // give the movie list its space
    for (user = 1; user < MAX_CUSTOMERS-1; user++) {
        for (int i=0; i<5; i++) 
            total += records[user].counts[i];
        records[user+1].movies = special_movies + total;
    }

    cout << "building lists of movies for each user" << endl;
    vector< vector<MovieRating> > temp_ratings(MAX_CUSTOMERS);
    for (int i=0; i<m_nRatingCount; i++) {
        rating = m_aRatings + i;
        movie = rating->MovieId;
        user = rating->CustId;
        MovieRating m;
        m.movie = movie;
        m.rating = rating->Rating;
        temp_ratings[user].push_back(m);
    }
    cout << "sorting" << endl;
    for (user = 1; user < MAX_CUSTOMERS; user++) {
        stable_sort(temp_ratings[user].begin(), temp_ratings[user].end(), sort_by_rating);
        for (int i=0; i<temp_ratings[user].size(); i++) 
            records[user].movies[i] = temp_ratings[user][i].movie;
    }
}

//
// ProcessFile
// - Load a history file in the format:
//
//   <MovieId>:
//   <CustomerId>,<Rating>
//   <CustomerId>,<Rating>
//   ...
//void Engine::ProcessFile(char *pwzFile)
void Engine::ProcessFile(char *pwzBuffer)
{
    FILE *stream;
    int custId, movieId, rating, pos = 0;
    
    cout << "Processing file: " << pwzBuffer << endl;

    if ((stream = fopen(pwzBuffer, "r")) == NULL) {
        cout << "error opening " << pwzBuffer << endl;
        exit(1);
    }

    // First line is the movie id
    fgets(pwzBuffer, 1000, stream);
    char *temp;
    temp = strtok(pwzBuffer, ":");
    movieId = atoi(temp);

    // Get all remaining rows
    fgets(pwzBuffer, 1000, stream);
    while ( !feof( stream ) )
    {
        pos = 0;
        temp = strtok(pwzBuffer, ",");
        custId = atoi(temp);
        temp = strtok(NULL, ",");
        rating = atoi(temp);
        
        m_aRatings[m_nRatingCount].MovieId = (short)movieId;
        m_aRatings[m_nRatingCount].CustId = custId;
        m_aRatings[m_nRatingCount].Rating = (BYTE)rating;
        m_aRatings[m_nRatingCount].Cache = 0;
        m_nRatingCount++;

        fgets(pwzBuffer, 1000, stream);
    }

    // Cleanup
    fclose( stream );
}


void Engine::ProcessTest(char *pwzFile) {
    FILE *streamIn, *streamOut;
    char pwzBuffer[1000];
    int custId, movieId, pos = 0;
    double rating;
    bool bMovieRow;

    sprintf(pwzBuffer, TEST_PATH, pwzFile);
    printf("\n\nProcessing test: %s\n", pwzBuffer);

    if ((streamIn = fopen(pwzBuffer, "r")) != 0) return;
    if ((streamOut = fopen(PREDICTION_FILE, "w")) != 0) return;

    fgets(pwzBuffer, 1000, streamIn);
    while ( !feof( streamIn ) )
    {
        bMovieRow = false;
        for (int i=0; i<(int)strlen(pwzBuffer); i++)
        {
            bMovieRow |= (pwzBuffer[i] == 58); 
        }

        pos = 0;
        if (bMovieRow)
        {
            //ParseInt(pwzBuffer, (int)strlen(pwzBuffer), pos, movieId);
            movieId = atoi(pwzBuffer);

            // Write same row to results
            fputs(pwzBuffer,streamOut); 
        }
        else
        {
            //ParseInt(pwzBuffer, (int)wcslen(pwzBuffer), pos, custId);
            custId = atoi(pwzBuffer);
            custId = m_mCustIds[custId];
            rating = PredictRating(movieId, custId);

            // Write predicted value
            sprintf(pwzBuffer,"%5.3f\n",rating);
            fputs(pwzBuffer,streamOut);
        }

        //wprintf(L"Got Line: %d %d %d \n", movieId, custId, rating);
        fgets(pwzBuffer, 1000, streamIn);
    }

    // Cleanup
    fclose( streamIn );
    fclose( streamOut );
}
