CXX = g++
VPATH = ./cpp/src
CXXFLAGS = -g -lsqlite3 -fopenmp -Wall -I$(VPATH) 

PROG = funk
OBJS = funk.o load.o optimizers.o predictor.o globals.h
UNAME := $(shell uname)
BONUS = -lblas
#BONUS = /home/sean/lib/atlas/lib/libcblas.a /home/sean/lib/atlas/lib/libatlas.a
ifeq ($(UNAME), Darwin)
	BONUS = -framework Accelerate 
endif
CRAP = ap.cpp alglibinternal.cpp alglibmisc.cpp linalg.cpp optimization.cpp 
CRAPOBJ = ${CRAP:.cpp=.o}

$(PROG): $(OBJS) $(CRAPOBJ) 
	$(CXX) $(CXXFLAGS) $(BONUS) $^ -o $(PROG) 
create_binaries: load.o globals.h
probe_bs: load.o  globals.h

optimizers.o: optimizers.h globals.h
predictor.o: predictor.h globals.h
load.o: load.h predictor.h globals.h

.PHONY: clean
clean:
	rm -f *.o *~ a.out core funk create_binaries probe_bs

.PHONY: all
all: clean create_binaries funk probe_bs
