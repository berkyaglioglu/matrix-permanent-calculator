all:
	nvcc -o perman main.cu -Xcompiler -fopenmp
