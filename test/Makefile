CC=nvcc

IDIR=inc

#OPENCV = `pkg-config opencv4 --libs --cflags` 
CFLAGS=-I $(IDIR) -I ../inc -O3 -lassimp -g

MAIN = test_runner

TEST = test.cu vec3_test.cu image_test.cu memory_manager_test.cu \
			 parser_test.cu triangle_manager_test.cu
TESTOBJECTS=$(patsubst %.cu,obj_test/%.o,$(TEST))

FILES = vec3.cu image.cu parser.cu config.cu
OBJECTS =$(patsubst %.cu,obj/%.o,$(FILES))

.PHONY = clean

all: MAIN
	@echo Compilation successful.

MAIN: $(TESTOBJECTS) $(OBJECTS)
	$(CC) obj/* obj_test/* test_runner.cu -o $(MAIN) $(CFLAGS) $(LIBS)

$(TESTOBJECTS): obj_test/%.o : src/%.cu
	$(CC) $(CFLAGS) -dlink -c $< -o $@

$(OBJECTS): obj/%.o : ../src/%.cu
	$(CC) $(CFLAGS) -dlink -c $< -o $@

clean:
	rm -f $(MAIN)
	rm -f obj/*.o
	rm -f obj_test/*.o
	rm -f imgs/*

