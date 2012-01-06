CXX = g++
INCLUDES =
CXXFLAGS = -g -Wall $(INCLUDES)
LDFLAGS = -g
LDLIBS = 

funk: load.o optimizers.o predictor.o
create_binaries: load.o

.PHONY: clean
clean:
	rm -f *.o *~ a.out core funk create_binaries

.PHONY: all
all: clean create_binaries funk
