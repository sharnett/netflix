#!/usr/bin/python
import numpy as np
from load_bs import *
from time import clock
from random import sample
from math import sqrt

num_movies = 17771
num_customers = 480189
#path = '/Users/srharnett/Downloads/download/'
path = '/home/sean/Documents/download/'

def stoch_grad_desc(data, num_features, max_iter):
    x = np.random.randn(num_movies, num_features)
    theta = np.random.randn(num_customers, num_features)
    for i in range(max_iter):
        [x, theta] = sweep(x, theta, data)
    return (x, theta)

def sweep(x, theta, data):
    tic = clock()
    print "doing SGD sweep..."
    alpha = .001
    total = 0
    for i, (rating, movie, user) in enumerate(data):
        prediction = np.dot(x[movie], theta[user])
        error = rating - prediction
        total += error*error
        temp = x[movie] 
        x[movie] += alpha*error*theta[user]
        theta[user] += alpha*error*temp
        if i%100000==0: print "processing rating: " + str(i)
    total /= len(data)
    print "time: " + str(clock() - tic) + " rmse: " + str(sqrt(total))
    return [x, theta]

def objective(x, theta, data):
    tic = clock()
    print "computing objective..."
    total = 0
    for i, (rating, movie, user) in enumerate(data):
        prediction = np.dot(x[movie], theta[user])
        error = rating - prediction
        total += error*error
        if i%100000==0: print "processing rating: " + str(i)
    total /= len(data)
    print "time: " + str(clock() - tic)
    return sqrt(total)

def main():
    training_size = 500000
    cv_size = training_size/5
    data = load_from_binary(path, 'data.txt')
    data = sample(data, training_size+cv_size)
    training_set = data[0:training_size]
    cv_set = data[training_size + 1:]
    (x, theta) = stoch_grad_desc(training_set, 1, 1)
    print "cross validation..."
    print "cost: " + str(objective(x, theta, cv_set))

if __name__ == "__main__":
    main()
