
loc?=strided3
bin?=bin

build:
	mpicc -o $(bin)/main $(loc)/main.c

run:
	mpirun -n 2 $(bin)/main

clean:
	rm -rf bin/*