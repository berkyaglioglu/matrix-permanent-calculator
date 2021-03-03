
#include <iostream>
#include <string>
#include <stdlib.h>
#include <omp.h>
#include <stdio.h>
#include "util.h"
#include "algo.h"
#include "algo.cu"
using namespace std;

enum algo {
  gpu_perman64_xlocal_algo = 1,
  gpu_perman64_xlocal_sparse_algo = 2,
  gpu_perman64_xshared_algo = 3,
  gpu_perman64_xshared_sparse_algo = 4,
  gpu_perman64_xshared_coalescing_algo =  5,
  gpu_perman64_xshared_coalescing_sparse_algo = 6,
  gpu_perman64_xshared_coalescing_mshared_algo = 7,
  gpu_perman64_xshared_coalescing_mshared_sparse_algo = 8,
  parallel_perman64_algo = 9,
  parallel_perman64_sparse_algo = 10,
  gpu_perman64_rasmussen_algo = 11,
  gpu_perman64_rasmussen_sparse_algo = 12,
  gpu_perman64_approximation_algo = 13,
  gpu_perman64_approximation_sparse_algo = 14,
  rasmussen_algo = 15,
  rasmussen_sparse_algo = 16,
  approximation_perman64_algo = 17,
  approximation_perman64_sparse_algo = 18,
  gpu_perman64_xglobal_algo = 19
};


template <class T>
void CallFunctions(T *mat, int *cptrs, int *rows, T *cvals, int *rptrs, int *cols, T *rvals, int dim, int nnz, int argc, char** argv) {
  int algo_id = atoi(argv[1]);
  double perman, start, end;
  int number_of_times, scale_intervals, scale_times, grid_dim, block_dim;

  switch(algo_id) {
    case gpu_perman64_xlocal_algo :
      if (argc != 4) {
        cout << "Number of parameters is wrong. Please check parameters by typing 'help'" << endl;
        break;
      }
      grid_dim = atoi(argv[2]);
      block_dim = atoi(argv[3]);
      start = omp_get_wtime();
      perman = gpu_perman64_xlocal(mat, dim, grid_dim, block_dim);
      end = omp_get_wtime();
      cout << "gpu_perman64_xlocal: " << perman << " in " << (end - start) << endl;
      break;

    case gpu_perman64_xlocal_sparse_algo :
      if (argc != 4) {
        cout << "Number of parameters is wrong. Please check parameters by typing 'help'" << endl;
        break;
      }
      grid_dim = atoi(argv[2]);
      block_dim = atoi(argv[3]);
      start = omp_get_wtime();
      perman = gpu_perman64_xlocal_sparse(mat, cptrs, rows, cvals, dim, grid_dim, block_dim);
      end = omp_get_wtime();
      cout << "gpu_perman64_xlocal_sparse: " << perman << " in " << (end - start) << endl;
      break;

    case gpu_perman64_xshared_algo :
      if (argc != 4) {
        cout << "Number of parameters is wrong. Please check parameters by typing 'help'" << endl;
        break;
      }
      grid_dim = atoi(argv[2]);
      block_dim = atoi(argv[3]);
      start = omp_get_wtime();
      perman = gpu_perman64_xshared(mat, dim, grid_dim, block_dim);
      end = omp_get_wtime();
      cout << "gpu_perman64_xshared: " << perman << " in " << (end - start) << endl;
      break;

    case gpu_perman64_xshared_sparse_algo :
      if (argc != 4) {
        cout << "Number of parameters is wrong. Please check parameters by typing 'help'" << endl;
        break;
      }
      grid_dim = atoi(argv[2]);
      block_dim = atoi(argv[3]);
      start = omp_get_wtime();
      perman = gpu_perman64_xshared_sparse(mat, cptrs, rows, cvals, dim, grid_dim, block_dim);
      end = omp_get_wtime();
      cout << "gpu_perman64_xshared_sparse: " << perman << " in " << (end - start) << endl;
      break;

    case gpu_perman64_xshared_coalescing_algo :
      if (argc != 4) {
        cout << "Number of parameters is wrong. Please check parameters by typing 'help'" << endl;
        break;
      }
      grid_dim = atoi(argv[2]);
      block_dim = atoi(argv[3]);
      start = omp_get_wtime();
      perman = gpu_perman64_xshared_coalescing(mat, dim, grid_dim, block_dim);
      end = omp_get_wtime();
      cout << "gpu_perman64_xshared_coalescing: " << perman << " in " << (end - start) << endl;
      break;

    case gpu_perman64_xshared_coalescing_sparse_algo :
      if (argc != 4) {
        cout << "Number of parameters is wrong. Please check parameters by typing 'help'" << endl;
        break;
      }
      grid_dim = atoi(argv[2]);
      block_dim = atoi(argv[3]);
      start = omp_get_wtime();
      perman = gpu_perman64_xshared_coalescing_sparse(mat, cptrs, rows, cvals, dim, grid_dim, block_dim);
      end = omp_get_wtime();
      cout << "gpu_perman64_xshared_coalescing_sparse: " << perman << " in " << (end - start) << endl;
      break;

    case gpu_perman64_xshared_coalescing_mshared_algo :
      if (argc != 4) {
        cout << "Number of parameters is wrong. Please check parameters by typing 'help'" << endl;
        break;
      }
      grid_dim = atoi(argv[2]);
      block_dim = atoi(argv[3]);
      start = omp_get_wtime();
      perman = gpu_perman64_xshared_coalescing_mshared(mat, dim, grid_dim, block_dim);
      end = omp_get_wtime();
      cout << "gpu_perman64_xshared_coalescing_mshared: " << perman << " in " << (end - start) << endl;
      break;

    case gpu_perman64_xshared_coalescing_mshared_sparse_algo :
      if (argc != 4) {
        cout << "Number of parameters is wrong. Please check parameters by typing 'help'" << endl;
        break;
      }
      grid_dim = atoi(argv[2]);
      block_dim = atoi(argv[3]);
      start = omp_get_wtime();
      perman = gpu_perman64_xshared_coalescing_mshared_sparse(mat, cptrs, rows, cvals, dim, grid_dim, block_dim);
      end = omp_get_wtime();
      cout << "gpu_perman64_xshared_coalescing_mshared_sparse: " << perman << " in " << (end - start) << endl;
      break;
    
    case parallel_perman64_algo :
      start = omp_get_wtime();
      perman = parallel_perman64(mat, dim);
      end = omp_get_wtime();
      cout << "parallel_perman64: " << perman << " in " << (end - start) << endl;
      break;

    case parallel_perman64_sparse_algo :
      start = omp_get_wtime();
      perman = parallel_perman64_sparse(mat, cptrs, rows, cvals, dim);
      end = omp_get_wtime();
      cout << "parallel_perman64_sparse: " << perman << " in " << (end - start) << endl;
      break;

    case gpu_perman64_rasmussen_algo :
      if (argc != 3) {
        cout << "Number of parameters is wrong. Please check parameters by typing 'help'" << endl;
        break;
      }
      number_of_times = atoi(argv[2]);
      start = omp_get_wtime();
      perman = gpu_perman64_rasmussen(mat, dim, number_of_times);
      end = omp_get_wtime();
      cout << "gpu_perman64_rasmussen: " << perman << " in " << (end - start) << endl;
      break;

    case gpu_perman64_rasmussen_sparse_algo :
      if (argc != 3) {
        cout << "Number of parameters is wrong. Please check parameters by typing 'help'" << endl;
        break;
      }
      number_of_times = atoi(argv[2]);
      start = omp_get_wtime();
      perman = gpu_perman64_rasmussen_sparse(rptrs, cols, dim, nnz, number_of_times);
      end = omp_get_wtime();
      cout << "gpu_perman64_rasmussen_sparse: " << perman << " in " << (end - start) << endl;
      break;

    case gpu_perman64_approximation_algo :
      if (argc != 5) {
        cout << "Number of parameters is wrong. Please check parameters by typing 'help'" << endl;
        break;
      }
      number_of_times = atoi(argv[2]);
      scale_intervals = atoi(argv[3]);
      scale_times = atoi(argv[4]);
      start = omp_get_wtime();
      perman = gpu_perman64_approximation(mat, dim, number_of_times, scale_intervals, scale_times);
      end = omp_get_wtime();
      cout << "gpu_perman64_approximation: " << perman << " in " << (end - start) << endl;
      break;

    case gpu_perman64_approximation_sparse_algo :
      if (argc != 5) {
        cout << "Number of parameters is wrong. Please check parameters by typing 'help'" << endl;
        break;
      }
      number_of_times = atoi(argv[2]);
      scale_intervals = atoi(argv[3]);
      scale_times = atoi(argv[4]);
      start = omp_get_wtime();
      perman = gpu_perman64_approximation_sparse(cptrs, rows, rptrs, cols, dim, nnz, number_of_times, scale_intervals, scale_times);
      end = omp_get_wtime();
      cout << "gpu_perman64_approximation_sparse: " << perman << " in " << (end - start) << endl;
      break;
    
    case rasmussen_algo :
      if (argc != 3) {
        cout << "Number of parameters is wrong. Please check parameters by typing 'help'" << endl;
        break;
      }
      number_of_times = atoi(argv[2]);
      start = omp_get_wtime();
      perman = rasmussen(mat, dim, number_of_times);
      end = omp_get_wtime();
      cout << "rasmussen: " << perman << " in " << (end - start) << endl;
      break;

    case rasmussen_sparse_algo :
      if (argc != 3) {
        cout << "Number of parameters is wrong. Please check parameters by typing 'help'" << endl;
        break;
      }
      number_of_times = atoi(argv[2]);
      start = omp_get_wtime();
      perman = rasmussen_sparse(cptrs, rows, rptrs, cols, dim, number_of_times);
      end = omp_get_wtime();
      cout << "rasmussen_sparse: " << perman << " in " << (end - start) << endl;
      break;

    case approximation_perman64_algo :
      if (argc != 5) {
        cout << "Number of parameters is wrong. Please check parameters by typing 'help'" << endl;
        break;
      }
      number_of_times = atoi(argv[2]);
      scale_intervals = atoi(argv[3]);
      scale_times = atoi(argv[4]);
      start = omp_get_wtime();
      perman = approximation_perman64(mat, dim, number_of_times, scale_intervals, scale_times);
      end = omp_get_wtime();
      cout << "approximation_perman64: " << perman << " in " << (end - start) << endl;
      break;

    case approximation_perman64_sparse_algo :
      if (argc != 5) {
        cout << "Number of parameters is wrong. Please check parameters by typing 'help'" << endl;
        break;
      }
      number_of_times = atoi(argv[2]);
      scale_intervals = atoi(argv[3]);
      scale_times = atoi(argv[4]);
      start = omp_get_wtime();
      perman = approximation_perman64_sparse(cptrs, rows, rptrs, cols, dim, number_of_times, scale_intervals, scale_times);
      end = omp_get_wtime();
      cout << "approximation_perman64_sparse: " << perman << " in " << (end - start) << endl;
      break;

    case gpu_perman64_xglobal_algo :
      if (argc != 4) {
        cout << "Number of parameters is wrong. Please check parameters by typing 'help'" << endl;
        break;
      }
      grid_dim = atoi(argv[2]);
      block_dim = atoi(argv[3]);
      start = omp_get_wtime();
      perman = gpu_perman64_xglobal(mat, dim, grid_dim, block_dim);
      end = omp_get_wtime();
      cout << "gpu_perman64_xglobal: " << perman << " in " << (end - start) << endl;
      break;

  }

}




int main(int argc, char** argv) {
  int algo_id = atoi(argv[1]);

  string type;
  cout << "Enter the type of the matrix(\"int\", \"float\", or \"double\"): " << endl;
  cin >> type;

  if (type == "int") {
    int *mat, *cvals, *rvals;
    int *cptrs, *rows, *rptrs, *cols;
    int dim, nnz;
    ReadMatrix(mat, dim, nnz);
    for (int i = 0; i < dim; i++) {
      for(int j = 0; j < dim; j++) {
        cout << mat[i*dim+j] << " ";
      }
      cout << endl;
    }
    
    matrix2compressed(mat, cptrs, rows, cvals, rptrs, cols, rvals, dim, nnz);
    
    CallFunctions(mat, cptrs, rows, cvals, rptrs, cols, rvals, dim, nnz, argc, argv);
    
    delete[] mat;
    delete[] cptrs;
    delete[] rows;
    delete[] cvals;
    delete[] rptrs;
    delete[] cols;
    delete[] rvals;
  }
  else if (type == "float") {
    float *mat, *cvals, *rvals;
    int *cptrs, *rows, *rptrs, *cols;
    int dim, nnz;
    ReadMatrix(mat, dim, nnz);
    for (int i = 0; i < dim; i++) {
      for(int j = 0; j < dim; j++) {
        cout << mat[i*dim+j] << " ";
      }
      cout << endl;
    }
    
    matrix2compressed(mat, cptrs, rows, cvals, rptrs, cols, rvals, dim, nnz);

    CallFunctions(mat, cptrs, rows, cvals, rptrs, cols, rvals, dim, nnz, argc, argv);
    
    delete[] mat;
    delete[] cptrs;
    delete[] rows;
    delete[] cvals;
    delete[] rptrs;
    delete[] cols;
    delete[] rvals;
  }
  else if (type == "double") {
    double *mat, *cvals, *rvals;
    int *cptrs, *rows, *rptrs, *cols;
    int dim, nnz;
    ReadMatrix(mat, dim, nnz);
    for (int i = 0; i < dim; i++) {
      for(int j = 0; j < dim; j++) {
        cout << mat[i*dim+j] << " ";
      }
      cout << endl;
    }
    
    matrix2compressed(mat, cptrs, rows, cvals, rptrs, cols, rvals, dim, nnz);

    CallFunctions(mat, cptrs, rows, cvals, rptrs, cols, rvals, dim, nnz, argc, argv);
    
    delete[] mat;
    delete[] cptrs;
    delete[] rows;
    delete[] cvals;
    delete[] rptrs;
    delete[] cols;
    delete[] rvals;
  }
  else {
    cout << "Value for \"type\" is nonexist" << endl;
    exit(1);
  }

  return 0;
}

