
#include <iostream>
#include <string>
#include <stdlib.h>
#include <omp.h>
#include <stdio.h>
#include "util.h"
#include "algo.h"
#include "algo.cu"

using namespace std;


template <class T>
void CallFunctions(T *mat, int *cptrs, int *rows, T *cvals, int *xadj, int *adj, T* val, int dim) {
  double perman, start, end;
/*
  start = omp_get_wtime();
  //perman = brute_w(xadj, adj, val, 2 * dim);
  end = omp_get_wtime();
  cout << "brute_w: " << perman << " in " << (end - start) << endl;

  start = omp_get_wtime();
  perman = perman64(mat, dim);
  end = omp_get_wtime();
  cout << "perman64: " << perman << " in " << (end - start) << endl;
*/
  start = omp_get_wtime();
  perman = parallel_perman64(mat, dim);
  end = omp_get_wtime();
  cout << "parallel_perman64: " << perman << " in " << (end - start) << endl;

  start = omp_get_wtime();
  perman = parallel_perman64_with_ccs(mat, dim, cptrs, rows, cvals);
  end = omp_get_wtime();
  cout << "parallel_perman64_with_ccs: " << perman << " in " << (end - start) << endl;

/*
  start = omp_get_wtime();
  perman = gpu_perman64_xlocal(mat, dim);
  end = omp_get_wtime();
  cout << "gpu_perman64_xlocal: " << perman << " in " << (end - start) << endl;

  start = omp_get_wtime();
  perman = gpu_perman64_xshared(mat, dim);
  end = omp_get_wtime();
  cout << "gpu_perman64_xshared: " << perman << " in " << (end - start) << endl;

  start = omp_get_wtime();
  perman = gpu_perman64_xshared_coalescing(mat, dim);
  end = omp_get_wtime();
  cout << "gpu_perman64_xshared_coalescing: " << perman << " in " << (end - start) << endl;

  start = omp_get_wtime();
  perman = gpu_perman64_xshared_coalescing_mshared(mat, dim);
  end = omp_get_wtime();
  cout << "gpu_perman64_xshared_coalescing_mshared: " << perman << " in " << (end - start) << endl;

  start = omp_get_wtime();
  perman = gpu_perman64_xshared_coalescing_with_ccs(mat, cptrs, rows, cvals, dim);
  end = omp_get_wtime();
  cout << "gpu_perman64_xshared_coalescing_with_ccs: " << perman << " in " << (end - start) << endl; 

  start = omp_get_wtime();
  perman = gpu_perman64_xshared_coalescing_mshared_with_ccs(mat, cptrs, rows, cvals, dim);
  end = omp_get_wtime();
  cout << "gpu_perman64_xshared_coalescing_mshared_with_ccs: " << perman << " in " << (end - start) << endl;

  */


  start = omp_get_wtime();
  perman = approximation_perman64(mat, dim);
  end = omp_get_wtime();
  cout << "approximation_perman64: " << perman << " in " << (end - start) << endl;


}



int main(int argc, char** argv) {
  int dim = -1;
  double density = -1;
  bool binary = false;
  string str_dim, str_density;

  for (int i = 1; i < argc; i++) {
    string arg = argv[i];
    if (arg == "-d") {
      i++;
      if (i < argc) {
        str_dim = argv[i];
        dim = atoi(argv[i]);
        if (dim <= 0) {
          cout << "Value for \"-d\" is invalid" << endl; 
          exit(1);
        }
      }
      else {
        cout << "Missing value for \"-d\"" << endl;
        exit(1);
      }
    }
    else if (arg == "-t") {
      i++;
      if (i < argc) {
        str_density = argv[i];
        density = atof(argv[i]);
        if (density <= 0) {
          cout << "Value for \"-t\" is invalid" << endl; 
          exit(1);
        }
      }
      else {
        cout << "Missing value for \"-t\"" << endl;
        exit(1);
      }
    }
    else if (arg == "-b") {
      binary = true;
    }
    else {
      cout << "Invalid parameters" << endl;
      exit(1);
    }
  }
  if(dim <= 0) {
    cout << "Value for \"-d\" is missing" << endl; 
    exit(1);
  }
  else if (density <= 0) {
    cout << "Value for \"-t\" is missing" << endl;
    exit(1);
  }

  string type;
  cout << "Enter the type of the matrix(\"int\", \"float\", or \"double\"): " << endl;
  cin >> type;

  string filename;
  if (binary) {
    filename = "sample/binary/" + str_dim + "_" + str_density + "_(1).txt";
  } else {
    filename = "sample/generic/" + str_dim + "_" + str_density + "_(1).txt";
  }

  if (type == "int") {
    //CreateMatrix(dim, density, binary);
    int *mat, *val, *cvals;
    ReadMatrix(mat, dim, filename);
    for (int i = 0; i < dim; i++) {
      for(int j = 0; j < dim; j++) {
        cout << mat[i*dim+j] << " ";
      }
      cout << endl;
    }
    int *xadj, *adj;//
    matrix2graph(mat, dim, xadj, adj, val);
    int *cptrs, *rows;
    matrix2CCS(mat, dim, cptrs, rows, cvals);
    CallFunctions(mat, cptrs, rows, cvals, xadj, adj, val, dim);
    
    delete[] mat;
    
    delete[] xadj;
    delete[] adj;
    delete[] val;

    delete[] cptrs;
    delete[] rows;
    delete[] cvals;
    
  }
  else if (type == "float") {
    //CreateMatrix(dim, density, binary);
    float *mat, *val, *cvals;
    ReadMatrix(mat, dim, filename);
    for (int i = 0; i < dim; i++) {
      for(int j = 0; j < dim; j++) {
        cout << mat[i*dim+j] << " ";
      }
      cout << endl;
    }
    int *xadj, *adj;
    matrix2graph(mat, dim, xadj, adj, val);
    int *cptrs, *rows;
    matrix2CCS(mat, dim, cptrs, rows, cvals);
    CallFunctions(mat, cptrs, rows, cvals, xadj, adj, val, dim);
    delete[] mat;

    delete[] xadj;
    delete[] adj;
    delete[] val;
    
    delete[] cptrs;
    delete[] rows;
    delete[] cvals;
  }
  else if (type == "double") {
    //CreateMatrix(dim, density, binary);
    double *mat, *val, *cvals;
    ReadMatrix(mat, dim, filename);
    for (int i = 0; i < dim; i++) {
      for(int j = 0; j < dim; j++) {
        cout << mat[i*dim+j] << " ";
      }
      cout << endl;
    }
    int *xadj, *adj;
    matrix2graph(mat, dim, xadj, adj, val);
    int *cptrs, *rows;
    matrix2CCS(mat, dim, cptrs, rows, cvals);
    CallFunctions(mat, cptrs, rows, cvals, xadj, adj, val, dim);
    delete[] mat;

    delete[] xadj;
    delete[] adj;
    delete[] val;

    delete[] cptrs;
    delete[] rows;
    delete[] cvals;
  }
  else {
    cout << "Value for \"type\" is nonexist" << endl;
    exit(1);
  }

  return 0;
}

