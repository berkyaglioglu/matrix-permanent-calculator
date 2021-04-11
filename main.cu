
#include <iostream>
#include <string>
#include <stdlib.h>
#include <ctype.h>
#include <unistd.h>
#include <omp.h>
#include <stdio.h>
#include "util.h"
#include "algo.h"
#include "algo.cu"
using namespace std;


template <class T>
void RunAlgo(T *mat, int *cptrs, int *rows, T *cvals, int *rptrs, int *cols, T *rvals, int perman_algo, int nov, int nnz, int gpu_num, int threads,
            bool gpu, bool cpu, bool dense, bool approximation, int number_of_times, int scale_intervals, int scale_times) 
{
  int grid_dim = 2048;
  int block_dim = 256;
  double start, end, perman;
  if (gpu) {
    if (dense) { // dense
      if (!approximation) { // exact
        if (perman_algo == 1) {
          start = omp_get_wtime();
          perman = gpu_perman64_xlocal(mat, nov, grid_dim, block_dim);
          end = omp_get_wtime();
          cout << "Result: gpu_perman64_xlocal " << perman << " in " << (end - start) << endl;
        } else if (perman_algo == 2) {
          start = omp_get_wtime();
          perman = gpu_perman64_xshared(mat, nov, grid_dim, block_dim);
          end = omp_get_wtime();
          cout << "Result: gpu_perman64_xshared " << perman << " in " << (end - start) << endl;
        } else if (perman_algo == 3) {
          start = omp_get_wtime();
          perman = gpu_perman64_xshared_coalescing(mat, nov, grid_dim, block_dim);
          end = omp_get_wtime();
          cout << "Result: gpu_perman64_xshared_coalescing " << perman << " in " << (end - start) << endl;
        } else if (perman_algo == 4) {
          start = omp_get_wtime();
          perman = gpu_perman64_xshared_coalescing_mshared(mat, nov, grid_dim, block_dim);
          end = omp_get_wtime();
          cout << "Result: gpu_perman64_xshared_coalescing_mshared " << perman << " in " << (end - start) << endl;
        } else if (perman_algo == 5) {
          start = omp_get_wtime();
          perman = gpu_perman64_xshared_coalescing_mshared_multigpu(mat, nov, gpu_num, grid_dim, block_dim);
          end = omp_get_wtime();
          cout << "Result: gpu_perman64_xshared_coalescing_mshared_multigpu " << perman << " in " << (end - start) << endl;
        } else if (perman_algo == 6) {
          start = omp_get_wtime();
          perman = gpu_perman64_xshared_coalescing_mshared_multigpu_manual_distribution(mat, nov, 4, grid_dim, block_dim);
          end = omp_get_wtime();
          cout << "Result: gpu_perman64_xshared_coalescing_mshared_multigpu_manual_distribution " << perman << " in " << (end - start) << endl;
        } else {
          cout << "Unknown Algorithm ID" << endl;
        } 
      } else { // approximation
        if (perman_algo == 1) { // rasmussen
          start = omp_get_wtime();
          perman = gpu_perman64_rasmussen(mat, nov, number_of_times);
          end = omp_get_wtime();
          printf("Result: gpu_perman64_rasmussen %2lf in %lf\n", perman, end-start);
        } else if (perman_algo == 2) { // approximation_with_scaling
          start = omp_get_wtime();
          perman = gpu_perman64_approximation(mat, nov, number_of_times, scale_intervals, scale_times);
          end = omp_get_wtime();
          printf("Result: gpu_perman64_approximation %2lf in %lf\n", perman, end-start);
        } else {
          cout << "Unknown Algorithm ID" << endl;
        } 
      }
    } else { // sparse
      if (!approximation) { // exact
        if (perman_algo == 1) {
          start = omp_get_wtime();
          perman = gpu_perman64_xlocal_sparse(mat, cptrs, rows, cvals, nov, grid_dim, block_dim);
          end = omp_get_wtime();
          cout << "Result: gpu_perman64_xlocal_sparse " << perman << " in " << (end - start) << endl;
        } else if (perman_algo == 2) {
          start = omp_get_wtime();
          perman = gpu_perman64_xshared_sparse(mat, cptrs, rows, cvals, nov, grid_dim, block_dim);
          end = omp_get_wtime();
          cout << "Result: gpu_perman64_xshared_sparse " << perman << " in " << (end - start) << endl;
        } else if (perman_algo == 3) {
          start = omp_get_wtime();
          perman = gpu_perman64_xshared_coalescing_sparse(mat, cptrs, rows, cvals, nov, grid_dim, block_dim);
          end = omp_get_wtime();
          cout << "Result: gpu_perman64_xshared_coalescing_sparse " << perman << " in " << (end - start) << endl;
        } else if (perman_algo == 4) {
          start = omp_get_wtime();
          perman = gpu_perman64_xshared_coalescing_mshared_sparse(mat, cptrs, rows, cvals, nov, grid_dim, block_dim);
          end = omp_get_wtime();
          cout << "Result: gpu_perman64_xshared_coalescing_mshared_sparse " << perman << " in " << (end - start) << endl;
        } else if (perman_algo == 5) {
          start = omp_get_wtime();
          perman = gpu_perman64_xshared_coalescing_mshared_multigpu_sparse(mat, cptrs, rows, cvals, nov, gpu_num, grid_dim, block_dim);
          end = omp_get_wtime();
          cout << "Result: gpu_perman64_xshared_coalescing_mshared_multigpu_sparse " << perman << " in " << (end - start) << endl;
        } else if (perman_algo == 6) {
          start = omp_get_wtime();
          perman = gpu_perman64_xshared_coalescing_mshared_multigpu_sparse_manual_distribution(mat, cptrs, rows, cvals, nov, 4, grid_dim, block_dim);
          end = omp_get_wtime();
          cout << "Result: gpu_perman64_xshared_coalescing_mshared_multigpu_sparse_manual_distribution " << perman << " in " << (end - start) << endl;
        } else {
          cout << "Unknown Algorithm ID" << endl;
        } 
      } else { // approximation
        if (perman_algo == 1) { // rasmussen
          start = omp_get_wtime();
          perman = gpu_perman64_rasmussen_sparse(rptrs, cols, nov, nnz, number_of_times);
          end = omp_get_wtime();
          printf("Result: gpu_perman64_rasmussen_sparse %2lf in %lf\n", perman, end-start);
        } else if (perman_algo == 2) { // approximation_with_scaling
          start = omp_get_wtime();
          perman = gpu_perman64_approximation_sparse(cptrs, rows, rptrs, cols, nov, nnz, number_of_times, scale_intervals, scale_times);
          end = omp_get_wtime();
          printf("Result: gpu_perman64_approximation_sparse %2lf in %lf\n", perman, end-start);
        } else {
          cout << "Unknown Algorithm ID" << endl;
        } 
      }
    }
  } else if (cpu) {
    if (dense) { // dense
      if (!approximation) { // exact
        start = omp_get_wtime();
        perman = parallel_perman64(mat, nov, threads);
        end = omp_get_wtime();
        cout << "Result: parallel_perman64 " << perman << " in " << (end - start) << endl;
      } else { // approximation
        if (perman_algo == 1) { // rasmussen
          start = omp_get_wtime();
          perman = rasmussen(mat, nov, number_of_times, threads);
          end = omp_get_wtime();
          printf("Result: rasmussen %2lf in %lf\n", perman, end-start);
        } else if (perman_algo == 2) { // approximation_with_scaling
          start = omp_get_wtime();
          perman = approximation_perman64(mat, nov, number_of_times, scale_intervals, scale_times, threads);
          end = omp_get_wtime();
          printf("Result: approximation_perman64 %2lf in %lf\n", perman, end-start);
        } else {
          cout << "Unknown Algorithm ID" << endl;
        } 
      }
    } else { // sparse
      if (!approximation) { // exact
        start = omp_get_wtime();
        perman = parallel_perman64_sparse(mat, cptrs, rows, cvals, nov, threads);
        end = omp_get_wtime();
        cout << "Result: parallel_perman64_sparse " << perman << " in " << (end - start) << endl;
      } else { // approximation
        if (perman_algo == 1) { // rasmussen
          start = omp_get_wtime();
          perman = rasmussen_sparse(cptrs, rows, rptrs, cols, nov, number_of_times, threads);
          end = omp_get_wtime();
          printf("Result: rasmussen_sparse %2lf in %lf\n", perman, end-start);
        } else if (perman_algo == 2) { // approximation_with_scaling
          start = omp_get_wtime();
          perman = approximation_perman64_sparse(cptrs, rows, rptrs, cols, nov, number_of_times, scale_intervals, scale_times, threads);
          end = omp_get_wtime();
          printf("Result: approximation_perman64_sparse %2lf in %lf\n", perman, end-start);
        } else {
          cout << "Unknown Algorithm ID" << endl;
        } 
      }
    }
  }
}


int main (int argc, char **argv)
{ 
  bool generic = true;
  bool dense = true;
  bool sortOrder = false;
  bool approximation = false;
  bool gpu = false;
  bool cpu = false;
  int gpu_num = 2;
  int threads = 16;
  string filename = "";
  int perman_algo = 1;

  int number_of_times = 100000;
  int scale_intervals = 4;
  int scale_times = 5;

  int c;

  while ((c = getopt (argc, argv, "bsrt:m:gd:cap:x:y:z:")) != -1)
    switch (c)
    {
      case 'b':
        generic = false;
        break;
      case 's':
        dense = false;
        break;
      case 'r':
        sortOrder = true;
        break;
      case 't':
        if (optarg[0] == '-'){
          fprintf (stderr, "Option -t requires an argument.\n");
          return 1;
        }
        threads = atoi(optarg);
        break;
      case 'm':
        if (optarg[0] == '-'){
          fprintf (stderr, "Option -m requires an argument.\n");
          return 1;
        }
        filename = optarg;
        break;
      case 'a':
        approximation = true;
        break;
      case 'g':
        gpu = true;
        break;
      case 'd':
        if (optarg[0] == '-'){
          fprintf (stderr, "Option -d requires an argument.\n");
          return 1;
        }
        gpu_num = atoi(optarg);
        break;
      case 'c':
        cpu = true;
        break;
      case 'p':
        if (optarg[0] == '-'){
          fprintf (stderr, "Option -p requires an argument.\n");
          return 1;
        }
        perman_algo = atoi(optarg);
        break;
      case 'x':
        if (optarg[0] == '-'){
          fprintf (stderr, "Option -x requires an argument.\n");
          return 1;
        }
        number_of_times = atoi(optarg);
        break;
      case 'y':
        if (optarg[0] == '-'){
          fprintf (stderr, "Option -y requires an argument.\n");
          return 1;
        }
        scale_intervals = atoi(optarg);
        break;
      case 'z':
        if (optarg[0] == '-'){
          fprintf (stderr, "Option -z requires an argument.\n");
          return 1;
        }
        scale_times = atoi(optarg);
        break;
      case '?':
        if (optopt == 't')
          fprintf (stderr, "Option -%c requires an argument.\n", optopt);
        else if (optopt == 'm')
          fprintf (stderr, "Option -%c requires an argument.\n", optopt);
        else if (optopt == 'd')
          fprintf (stderr, "Option -%c requires an argument.\n", optopt);
        else if (optopt == 'p')
          fprintf (stderr, "Option -%c requires an argument.\n", optopt);
        else if (isprint (optopt))
          fprintf (stderr, "Unknown option `-%c'.\n", optopt);
        else
          fprintf (stderr,
                   "Unknown option character `\\x%x'.\n",
                   optopt);
        return 1;
      default:
        abort ();
    }

  if (filename == "") {
    fprintf (stderr, "Option -m is a required argument.\n");
  }

  for (int index = optind; index < argc; index++)
  {
    printf ("Non-option argument %s\n", argv[index]);
  }

  if (!cpu && !gpu) {
    gpu = true;
  }


  int nov, nnz;
  string type;

  ifstream inFile(filename);
  string line;
  getline(inFile, line);
  istringstream iss(line);
  iss >> nov >> nnz >> type;

  if (type == "int") {
    int* mat = new int[nov*nov];
    ReadMatrix(mat, inFile, nov, generic);

    int *cvals, *rvals;
    int *cptrs, *rows, *rptrs, *cols;
    if (sortOrder) {
      matrix2compressed_sortOrder(mat, cptrs, rows, cvals, rptrs, cols, rvals, nov, nnz);
    } else {
      matrix2compressed(mat, cptrs, rows, cvals, rptrs, cols, rvals, nov, nnz);
    }
    
    for (int i = 0; i < nov; i++) {
      for(int j = 0; j < nov; j++) {
        cout << mat[i*nov+j] << " ";
      }
      cout << endl;
    }

    RunAlgo(mat, cptrs, rows, cvals, rptrs, cols, rvals, perman_algo, nov, nnz, gpu_num, threads, gpu, cpu, dense, approximation, number_of_times, scale_intervals, scale_times);

    delete[] mat;
    delete[] cptrs;
    delete[] rows;
    delete[] cvals;
    delete[] rptrs;
    delete[] cols;
    delete[] rvals;
    
  } else if (type == "float") {
    float* mat = new float[nov*nov];
    ReadMatrix(mat, inFile, nov, generic);

    float *cvals, *rvals;
    int *cptrs, *rows, *rptrs, *cols;
    if (sortOrder) {
      matrix2compressed_sortOrder(mat, cptrs, rows, cvals, rptrs, cols, rvals, nov, nnz);
    } else {
      matrix2compressed(mat, cptrs, rows, cvals, rptrs, cols, rvals, nov, nnz);
    }
    
    for (int i = 0; i < nov; i++) {
      for(int j = 0; j < nov; j++) {
        if (mat[i*nov+j] == 0) {
          cout << "0.0 ";
        } else {
          cout << to_string(mat[i*nov+j]).substr(0,3) << " ";
        }
      }
      cout << endl;
    }

    RunAlgo(mat, cptrs, rows, cvals, rptrs, cols, rvals, perman_algo, nov, nnz, gpu_num, threads, gpu, cpu, dense, approximation, number_of_times, scale_intervals, scale_times);

    delete[] mat;
    delete[] cptrs;
    delete[] rows;
    delete[] cvals;
    delete[] rptrs;
    delete[] cols;
    delete[] rvals;

  } else if (type == "double") {
    double* mat = new double[nov*nov];
    ReadMatrix(mat, inFile, nov, generic);

    double *cvals, *rvals;
    int *cptrs, *rows, *rptrs, *cols;
    if (sortOrder) {
      matrix2compressed_sortOrder(mat, cptrs, rows, cvals, rptrs, cols, rvals, nov, nnz);
    } else {
      matrix2compressed(mat, cptrs, rows, cvals, rptrs, cols, rvals, nov, nnz);
    }
    
    for (int i = 0; i < nov; i++) {
      for(int j = 0; j < nov; j++) {
        if (mat[i*nov+j] == 0) {
          cout << "0.0 ";
        } else {
          cout << to_string(mat[i*nov+j]).substr(0,3) << " ";
        }
      }
      cout << endl;
    }

    RunAlgo(mat, cptrs, rows, cvals, rptrs, cols, rvals, perman_algo, nov, nnz, gpu_num, threads, gpu, cpu, dense, approximation, number_of_times, scale_intervals, scale_times);

    delete[] mat;
    delete[] cptrs;
    delete[] rows;
    delete[] cvals;
    delete[] rptrs;
    delete[] cols;
    delete[] rvals;

  }

  return 0;
}

