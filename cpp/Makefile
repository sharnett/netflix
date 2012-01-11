CXX = g++
INCLUDES = 
CXXFLAGS = -g -Wall $(INCLUDES)

PROG = funk
OBJS = funk.o load.o optimizers.o predictor.o globals.h
UNAME := $(shell uname)
BONUS = -lblas
ifeq ($(UNAME), Darwin)
	BONUS = -framework Accelerate 
endif

$(PROG): $(OBJS) 
	$(CXX) $(CXXFLAGS) $(OBJS) $(BONUS) -o $(PROG)
create_binaries: load.o globals.h
probe_bs: load.o  globals.h

optimizers.o: optimizers.h globals.h
predictor.o: predictor.h globals.h
load.o: load.h globals.h

.PHONY: clean
clean:
	rm -f *.o *~ a.out core funk create_binaries probe_bs

.PHONY: all
all: clean create_binaries funk probe_bs
