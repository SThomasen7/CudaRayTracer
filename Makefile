CC=nvcc

IDIR=inc
SRCDIR=src
ODIR=obj

#OPENCV = `pkg-config opencv4 --libs --cflags` 
#CFLAGS=-I $(IDIR) -O0 -g -lassimp -G
CFLAGS=-I $(IDIR) -O3 -g -lassimp

MAIN = raytracer

RENDER =
RENDEROBJECTS=$(patsubst %.cpp,obj/%.o,$(RENDER))
CRENDER = ray_cuda_headers.cu  lights.cu \
					sphere.cu camera.cu vec3.cu image.cu \
					hit.cu ray_tracing.cu render_engine.cu \
					material_mem.cu config.cu parser.cu \
					asset.cu triangle.cu main.cu 
CRENDEROBJECTS=$(patsubst %.cu,obj/%.o,$(CRENDER))

.PHONY = clean

all: MAIN
	@echo Compilation successful.

debug: CFLAGS=-I $(IDIR) -O0 -G -g -lassimp
debug: MAIN

MAIN: $(RENDEROBJECTS) $(CRENDEROBJECTS)
	$(CC) $(ODIR)/*.o -o $(MAIN) $(CFLAGS) $(LIBS)

$(RENDEROBJECTS): obj/%.o : src/%.cpp
	$(CC) $(CFLAGS) -c $< -o $@

$(CRENDEROBJECTS): obj/%.o : src/%.cu
	$(CC) $(CFLAGS) -dlink -c $< -o $@

clean:
	rm -f $(MAIN)
	rm -f $(ODIR)/*

