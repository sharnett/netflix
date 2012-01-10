#ifndef load_H
#define load_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <map>
#include "globals.h"

using namespace std;

int LoadHistory(Data *ratings, bool dump_dict = 0);
void ProcessFile(char *history_file, Data *ratings, int& num_ratings, 
        map<int, int>& user_dict);
void DumpBinary(Data *ratings, int num_ratings, string filename = "cpp/binary.txt");
int LoadBinary(Data *ratings, string filename = "cpp/binary.txt");
void dump_avg(Data *ratings, int num_ratings);
void load_avg(float *movie_avg);
void load_user_dict(map<int, int>& user_dict);
string get_data_folder();

#endif
