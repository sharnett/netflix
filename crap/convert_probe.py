#!/usr/bin/python
from time import clock
from load_bs import dump_data, load_user_dict

#path = '/Users/srharnett/Downloads/download/'
path = '/home/sean/Documents/download/'

def convert_probe(path, data):
    tic = clock()
    infile = open(path + 'probe.txt', 'r')
    movies = {}
    user_dict = load_user_dict(path, 'user_dict.txt')
    num_total = 100480507
    num_probe = 0

    print "loading probe.txt..."
    for line in infile:
        if line.find(':') > 0:
            movie = int(line[:-2])
            movies[movie] = []
            continue
        user = user_dict[int(line)]
        movies[movie].append(user)
        num_probe += 1
    infile.close()
    total = 0

    num_non_probe = num_total - num_probe
    dt = np.dtype={'names': ['rating','movie','user'], 'formats': ['u1','u2','u4']}
    probe = np.zeros((num_probe,), dt)
    non_probe = np.zeros((num_non_probe,), dt)

    print "adding ratings..."
    i=0
    for movie in range(1, 17771):
        if movie%100==0: print "processing movie: " + str(movie)
        users = movies.get(movie, [])
        while data[i]['movie'] != movie:
            i += 1
        while data[i]['movie'] == movie:
            try:
                users.find(data[i]['user'])
                rating = data[i]['rating']
            except ValueError:



            while ratings.customers[i] != customer:
                value = ratings.values[i]
                non_probe.add(movie, ratings.customers[i], value)
                i += 1
                n += 1
            value = ratings.values[i]
            probe.add(movie, customer, value)
            i += 1
            p += 1
    print "probes: %d non-probes: %d" % (p, n)
    print "dumping..."
#dump_data(path, filename, data)
    dump_binary(path, [], [], probe, 'probe')
    dump_binary(path, [], [], non_probe, 'non_probe')
    print "total time: " + str(clock() - tic)

convert_probe([], path)
