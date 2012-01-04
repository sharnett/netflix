#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <map>

using namespace std;

typedef unsigned char BYTE;

struct Data {
    int user;
    short movie;
    BYTE rating;
//    float Cache;
};

int LoadHistory(Data *ratings);
void ProcessFile(char *history_file, Data *ratings, int& num_ratings, map<int, int>& user_map);
void DumpBinary(Data *ratings, int num_ratings);
int LoadBinary(Data *ratings);
string get_data_folder();
