# matrix-permanent-calculator
Parallel matrix permanent calculation in both CPU and GPU.

# Experiments

## Load modules
module load gcc/5.3.0
module load cuda/10.0

## Compilation
Compile the code with the following command.
  * nvcc main.cu -O3 -Xcompiler -fopenmp

Then, set number of threads to be used in CPU.
  * export OMP_NUM_THREADS=32

In order to run, specify parameters, such that "-d" is dimension of the matrix, "-t" is density of the matrix, and "-b"(optional) is for making the matrix binary.
  * ./a.out -d 32 -t 0.15 -b

Then, user is prompted "Enter the type of the matrix("int", "float", or "double"):". Enter the type for the matrix here. 

