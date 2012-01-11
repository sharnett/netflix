#ifndef load_H
#define load_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <fstream>
#include <map>
#include "globals.h"

using namespace std;

int load_history(Data *ratings, bool dump_dict = 0);
void process_file(char *history_file, Data *ratings, int& num_ratings, 
        map<int, int>& user_dict);
void dump_binary(Data *ratings, int num_ratings, string filename = "cpp/binary.txt");
int load_binary(Data *ratings, string filename = "cpp/binary.txt");
void dump_avg(Data *ratings, int num_ratings);
void load_avg(float *movie_avg);
void load_user_dict(map<int, int>& user_dict);
string get_data_folder();

#endif
