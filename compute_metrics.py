#!/usr/bin/python
from time import clock

num_movies = 17770
num_customers = 480189
path = '/Users/srharnett/Downloads/download/'

class Movie:
    def __init__(self):
        self.rating_count = 0
        self.rating_sum = 0
        self.rating_avg = 0
        self.pseudo_avg = 0
    def compute_avg(self):
        self.pseudo_avg = (3.23 * 25 + self.rating_sum) / (25.0 + self.rating_count)
        self.rating_avg = float(self.rating_sum) / self.rating_count
    def add_rating(self, rating):
        self.rating_count += 1
        self.rating_sum += rating
    def __str__(self):
        return "rating count: %d, rating sum: %d, avg: %.2g, pavg: %.2g" % \
                (self.rating_count, self.rating_sum, self.rating_avg, self.pseudo_avg)


class User:
    def __init__(self, id):
        self.id = id
        self.rating_count = 0
        self.rating_sum = 0
    def add_rating(self, rating):
        self.rating_count += 1
        self.rating_sum += rating
    def __str__(self):
        return "customer id: %d, rating count: %d, rating sum: %d" % \
                (self.id, self.rating_count, self.rating_sum)


def compute_metrics(data):
    tic = clock()
    print "computing metrics..."
    i = 0
    # movie IDs start at 1, so we keepy a dummy movie at the beginning
    movies = [-1]*(num_movies+1)
    users = [-1]*num_users
    for (rating, movie, user) in data:
        if i%100000==0: print "processing rating: " + str(i)
        if movies[movie] == -1:
            movies[movie] = Movie()
        movies[movie].add_rating(rating)
        if users[user] == -1:
            users[user] = User(user)
        users[user].add_rating(rating)
        i += 1
    print "time: " + str(clock() - tic)
    return [movies, users]
