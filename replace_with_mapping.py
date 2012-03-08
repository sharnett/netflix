#!/usr/bin/python
from time import clock
from load_bs import *
from array import array
import numpy as np

path = '/Users/srharnett/Downloads/download/'

def replace_with_mapping(ratings, customer_dict, path, prefix):
    tic = clock()
    print "replacing..."
    temp = [customer_dict[customer] for customer in ratings.customers]
    ratings.customers = array('i')
    ratings.customers.fromlist(temp)
    print "dumping..."
    f = open(path + prefix + '_c.txt', 'wb')
    ratings.customers.tofile(f)
    f.close()
    print "total time: " + str(clock() - tic)

def create_struct_array(ratings, path, prefix):
    tic = clock()
    print "doing it"
    n = len(ratings.movies)
    struct_array = np.zeros((n,), \
        dtype={'names': ['rating','movie','user'], 'formats': ['u1','u2','u4']})
    for i in xrange(n):
        struct_array[i]['rating'] = ratings.values[i]
        struct_array[i]['movie'] = ratings.movies[i]
        struct_array[i]['user'] = ratings.customers[i]
        if i%100000==0: print "processing rating: " + str(i)
    f = open(path + prefix + '.txt', 'wb')
    struct_array.tofile(f)
    f.close()
    print "total time: " + str(clock() - tic)

if __name__ == "__main__":
    num_ratings = 99072017
    [movies, customers, ratings] = load_from_binary(path, num_ratings)
    #customer_dict = create_customer_dict(ratings.customers, path)
    #prefix = "new_non_probe"
    #replace_with_mapping(ratings, customer_dict, path, prefix)
    prefix = "struct_array"
    create_struct_array(ratings, path, prefix)
