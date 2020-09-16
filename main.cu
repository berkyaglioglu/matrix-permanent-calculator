
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
void CallFunctions(T *mat, int *xadj, int *adj, T* val, int dim) {
  unsigned long long int perman;
  double start, end;
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

/*
  start = omp_get_wtime();
  perman = gpu_perman64_warplevel(mat, dim);
  end = omp_get_wtime();
  cout << "gpu_perman64_warplevel: " << perman << " in " << (end - start) << endl;

  start = omp_get_wtime();
  perman = sparser_perman64_w(xadj, adj, val, dim); 
  end = omp_get_wtime();
  cout << "sparser_perman64_w: " << perman << " in " << (end - start) << endl;

  start = omp_get_wtime();
  perman = sparser_skip_perman64_w(xadj, adj, val, mat, dim);
  end = omp_get_wtime();
  cout << "sparser_skip_perman64_w: " << perman << " in " << (end - start) << endl;

  start = omp_get_wtime();
  perman = parallel_skip_perman64_w(xadj, adj, val, mat, dim);
  end = omp_get_wtime();
  cout << "parallel_skip_perman64_w: " << perman << " in " << (end - start) << endl;

  start = omp_get_wtime();
  perman = parallel_skip_perman64_w_balanced(xadj, adj, val, mat, dim);
  end = omp_get_wtime();
  cout << "parallel_skip_perman64_w_balanced: " << perman << " in " << (end - start) << endl;     
*/
}



int main(int argc, char** argv) {
	int dim = -1;
	double density = -1;
	bool binary = false;

	for (int i = 1; i < argc; i++) {
		string arg = argv[i];
		if (arg == "-d") {
			i++;
			if (i < argc) {
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

  if (type == "int") {
    CreateMatrix(dim, density, binary);
    int *mat, *val;
    ReadMatrix(mat, dim);
    for (int i = 0; i < dim; i++) {
      for(int j = 0; j < dim; j++) {
        cout << mat[i*dim+j] << " ";
      }
      cout << endl;
    }
    int *xadj, *adj;
    matrix2graph(mat, dim, xadj, adj, val);
    CallFunctions(mat, xadj, adj, val, dim);
    delete[] mat;
    delete[] xadj;
    delete[] adj;
    delete[] val;
  }
  else if (type == "float") {
    CreateMatrix(dim, density, binary);
    float *mat, *val;
    ReadMatrix(mat, dim);
    for (int i = 0; i < dim; i++) {
      for(int j = 0; j < dim; j++) {
        cout << mat[i*dim+j] << " ";
      }
      cout << endl;
    }
    int *xadj, *adj;
    matrix2graph(mat, dim, xadj, adj, val);
    CallFunctions(mat, xadj, adj, val, dim);
    delete[] mat;
    delete[] xadj;
    delete[] adj;
    delete[] val;
  }
  else if (type == "double") {
    CreateMatrix(dim, density, binary);
    double *mat, *val;
    ReadMatrix(mat, dim);
    for (int i = 0; i < dim; i++) {
      for(int j = 0; j < dim; j++) {
        cout << mat[i*dim+j] << " ";
      }
      cout << endl;
    }
    int *xadj, *adj;
    matrix2graph(mat, dim, xadj, adj, val);
    CallFunctions(mat, xadj, adj, val, dim);
    delete[] mat;
    delete[] xadj;
    delete[] adj;
    delete[] val;
  }
  else {
    cout << "Value for \"type\" is nonexist" << endl;
    exit(1);
  }

	return 0;
}

