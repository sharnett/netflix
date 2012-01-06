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

int LoadHistory(Data *ratings);
void ProcessFile(char *history_file, Data *ratings, int& num_ratings, map<int, int>& user_map);
void DumpBinary(Data *ratings, int num_ratings, string filename = "cpp/binary.txt");
int LoadBinary(Data *ratings, string filename = "cpp/binary.txt");
void dump_avg(Data *ratings, int num_ratings);
void load_avg(float *movie_avg);
string get_data_folder();

#endif
