CXX = g++
INCLUDES =
CXXFLAGS = -g -Wall $(INCLUDES)
LDFLAGS = -g -framework Accelerate
LDLIBS = 

funk: load.o optimizers.o predictor.o
create_binaries: load.o
probe_bs: load.o

.PHONY: clean
clean:
	rm -f *.o *~ a.out core funk create_binaries probe_bs

.PHONY: all
all: clean create_binaries funk probe_bs
