#!/usr/bin/python
from cPickle import dump, load
from os import listdir
from string import strip
from time import clock
from random import shuffle
import numpy as np

num_movies = 17770
num_users = 480189
#path = '/Users/srharnett/Downloads/download/'
path = '/home/sean/Documents/download/'

def load_from_original_text(path):
    tic = clock()
    print "loading files..."
    listing = listdir(path)
    listing.sort()
    num_ratings = 100480507
    i=0
    j=0
    # this is the main data structure, should turn out to be around 700 MB
    data = np.zeros((num_ratings,), \
        dtype={'names': ['rating','movie','user'], 'formats': ['u1','u2','u4']})
    # this is for creating the user dictionary
    # the user IDs are not given in a compact 0-n numbering, but instead
    # bounce all over the place with huge gaps, making this necessary
    shuffled_users = range(num_users)
    shuffle(shuffled_users)
    user_dict = {}

    # read data from each file
    for file_number, infile in enumerate(listing):
        if file_number%100==0: print "processing file: " + infile
        f = open(path + infile)
        # first line is the movie id
        movie = int(f.readline().strip().strip(':'))
        # subsequent lines are user_id, rating, date
        for line in f:
            [user, rating, d] = line.split(',')
            data[i]['rating'] = rating
            data[i]['movie'] = movie
            # if new user, assign a compact id and add to dictionary
            if not user in user_dict:
                user_dict[user] = shuffled_users[j]
                j += 1
            data[i]['user'] = user_dict[user]
            i += 1
        f.close()
    
    print "loading time: " + str(clock() - tic)
    return [data, user_dict]


def dump_data(path, filename, data):
    tic = clock()
    print "dumping data..."
    f = open(path + filename, 'wb')
    data.tofile(f)
    f.close()
    print "dumping time: " + str(clock() - tic)


def load_from_binary(path, filename):
    print "loading data..."
    tic = clock()
    dt = np.dtype={'names': ['rating','movie','user'], 'formats': ['u1','u2','u4']}
    f = open(path + filename, 'rb')
    data = np.fromfile(f, dt)
    f.close()
    print "loading time: " + str(clock() - tic)
    print "number of ratings: " + str(len(data))
    return data


def dump_user_dict(path, filename, user_dict):
    f = open(path + filename, 'wb')
    dump(user_dict, f)
    f.close()

def load_user_dict(path, filename):
    f = open(path + filename, 'rb')
    user_dict = load(f)
    f.close()
    return user_dict

if __name__ == "__main__":
    [data, user_dict] = load_from_original_text(path + 'training_set/')
    dump_data(path, 'data.txt', data)
    dump_user_dict(path, 'user_dict.txt', user_dict)
#    data = load_from_binary(path, 'data.txt')
#    user_dict = load_user_dict(path, 'user_dict.txt')
