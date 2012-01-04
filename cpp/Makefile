CXX = g++
INCLUDES =
CXXFLAGS = -g -Wall $(INCLUDES)
LDFLAGS = -g
LDLIBS = 

funk: load.o 

.PHONY: clean
clean:
	rm -f *.o *~ a.out core funk

.PHONY: all
all: clean funk
