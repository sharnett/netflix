#include "load.h"

using namespace std;

int LoadHistory(Data *ratings) {
    time_t start,end; time(&start);
    string data_folder = get_data_folder() + "training_set/";
    char data_file[100];
    int num_ratings = 0;
    map<int, int> user_map; // Map for one time translation of ids to compact array index
    for (int i = 1; i <= 17770; i++) {
        sprintf(data_file, "%smv_00%05d.txt", data_folder.c_str(), i);
        ProcessFile(data_file, ratings, num_ratings, user_map);
    }
    time(&end);
    cout << "time: " << difftime(end,start) << "s" << endl;
    return num_ratings;
}

// - Load a history file in the format:
//   <MovieId>:
//   <CustomerId>,<Rating>
//   <CustomerId>,<Rating>
void ProcessFile(char *history_file, Data *ratings, int& num_ratings, map<int, int>& user_map) {
    cout << "Processing file: " << history_file << endl;
    FILE *stream;
    if ((stream = fopen(history_file, "r")) == NULL) {
        cout << "error opening " << history_file << endl;
        exit(1);
    }

    int user; short movie; 
    char buf[1000];

    // First line is the movie id
    fgets(buf, 1000, stream);
    char *temp;
    temp = strtok(buf, ":");
    movie = (short)atoi(temp);

    Data *datum;
    map<int, int>::iterator itr;
    int cid;

    // Get all remaining rows
    while (fgets(buf, 1000, stream)) {
        datum = &ratings[num_ratings++];
        datum->movie = movie;
//        datum->Cache = 0;
        temp = strtok(buf, ",");
        user = atoi(temp);
        temp = strtok(NULL, ",");
        datum->rating = (BYTE)atoi(temp);

        itr = user_map.find(user);
        if (itr == user_map.end()) {
            cid = (int)user_map.size();
            user_map[user] = cid;
        }
        else
            cid = itr->second;
        datum->user = cid;
    }
    fclose(stream);
}

void DumpBinary(Data *ratings, int num_ratings, string filename) {
    string filepath = get_data_folder() + filename;
    cout << filepath << endl;
    FILE* f = fopen(filepath.c_str(), "w");
    fwrite(ratings, sizeof(Data), num_ratings, f);
    fclose(f);
}

int LoadBinary(Data *ratings, string filename) {
    string filepath = get_data_folder() + filename;
    FILE* f = fopen(filepath.c_str(), "r");
    int num_ratings = fread(ratings, sizeof(Data), 100480507, f);
    fclose(f);
    cout << num_ratings << " ratings loaded" << endl;
    return num_ratings;
}

string get_data_folder() {
    ifstream f("data_folder.txt");
    string df;
    getline(f, df);
    f.close();
    return df;
}
