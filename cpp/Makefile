CXX = g++
VPATH = ./cpp/src
CXXFLAGS = -g -lsqlite3 -fopenmp -Wall -I$(VPATH) 

PROG = funk
OBJS = funk.o load.o optimizers.o predictor.o globals.h
PROG2 = best_lambda
OBJS2 = best_lambda.o load.o optimizers.o predictor.o fmin.o globals.h
UNAME := $(shell uname)
#BONUS = -lblas
#BONUS = /home/sean/lib/atlas/lib/libcblas.a /home/sean/lib/atlas/lib/libatlas.a
BONUS = -L/home/sean/bin/intel/composer_xe_2011_sp1.7.256/composer_xe_2011_sp1.7.256/mkl/lib/intel64 -lmkl_rt
ifeq ($(UNAME), Darwin)
	BONUS = -framework Accelerate 
endif
CRAP = ap.cpp alglibinternal.cpp alglibmisc.cpp linalg.cpp optimization.cpp 
CRAPOBJ = ${CRAP:.cpp=.o}

$(PROG): $(OBJS) $(CRAPOBJ) 
	$(CXX) $(CXXFLAGS) $(BONUS) $^ -o $(PROG) 
create_binaries: load.o globals.h
probe_bs: load.o  globals.h
best_lambda: $(OBJS2) $(CRAPOBJ) 
	$(CXX) $(CXXFLAGS) $(BONUS) $^ -o $(PROG2) 
find_average: load.o globals.h
best_averages: best_averages.o $(CRAPOBJ) load.o fmin.o globals.h
	$(CXX) $(CXXFLAGS) $(BONUS) $^ -o best_averages
train_and_dump_avgs: train_and_dump_avgs.o $(CRAPOBJ) load.o globals.h
	$(CXX) $(CXXFLAGS) $(BONUS) $^ -o train_and_dump_avgs

optimizers.o: optimizers.h globals.h
predictor.o: predictor.h globals.h
load.o: load.h predictor.h globals.h
fmin.o: fmin.h

PROGS = funk create_binaries probe_bs best_lambda find_average best_averages train_and_dump_avgs

.PHONY: clean
clean:
	rm -f *.o *~ a.out core $(PROGS)

.PHONY: all
all: clean create_binaries funk probe_bs
