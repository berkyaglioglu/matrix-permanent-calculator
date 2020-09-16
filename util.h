#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <ctime>
#include <cassert>
using namespace std;

string seperator = "************************************************************************";


void CreateMatrix(int dim, double density, bool binary) {
	ofstream outFile("matrix.txt");
	if (binary) {
		outFile << dim << " " << "binary" << endl;
	}
	else {
		outFile << dim << " " << "generic" << endl;
	}

	srand(time(0));
	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {
			if ((rand() % 100) < (100 * density)) {
				if (binary) {
					outFile << i << " " << j << " " << 1 << endl;
				}
				else {
					outFile << i << " " << j << " " << (rand() % 10 + 1) << endl;
				}
			}
			else {
				outFile << i << " " << j << " " << 0 << endl;
			}
		}
	}

	outFile.close();
}

template <class T>
void ReadMatrix(T* & mat, int dim) {
	ifstream inFile("matrix.txt");
	mat = new T[dim * dim];
	int* row_degs = new int[dim];
	int* col_degs = new int[dim];
	
	int i, j, val;
	string line;
	int nonzeroNum = 0;
	while (getline(inFile, line)) {
		istringstream iss(line);
		if (!(iss >> i >> j >> val)) { continue; } // erroneous line
		if (val != 0) {
			mat[i * dim + j] = val; row_degs[i]++; col_degs[j]++;
			nonzeroNum++;
		}	
	}
	
	int zero_deg = 0;
	int one_deg = 0;
	int two_deg = 0;
	for(int i = 0; i < dim; i++) {
		if(row_degs[i] == 0) {
			zero_deg++;
			cout << "Row " << i << " has no nonzeros" << endl;
		} else if(row_degs[i] == 1) {
			one_deg++;
		} else if(row_degs[i] == 2) {
			two_deg++;
		}

		if(col_degs[i] == 0) {
			zero_deg++;
			cout << "Col " << i << " has no nonzeros" << endl;
		} else if(col_degs[i] == 1) {
			one_deg++;
		} else if(col_degs[i] == 2) {
			two_deg++;
		}
	}
	delete [] row_degs;
	delete [] col_degs;
  
	if(zero_deg > 0) {
		cout << "Exiting due to non-empty rows/columns " << endl;
		exit(1);
	} else {
		cout << "Number of rows/cols is " << dim << endl;
		cout << "Number of nonzeros is " << nonzeroNum << endl;
		cout << "#d1 vertices: " << one_deg << endl;
		cout << "#d2 vertices: " << two_deg << endl;
	}
	cout << seperator << endl;
}


template <class T>
void matrix2graph(T* mat, int nov, int*& xadj, int*& adj, T*& val) {
	int nnz = 0;
	for(int i = 0; i < nov * nov; i++) {
		assert(mat[i] >= 0);
		if(mat[i] > 0) {
			nnz++;
		}
	}

	xadj = new int[(2 * nov) + 1];
	adj = new int[2 * nnz];
	val = new T[2 * nnz];
  
	nnz = 0;
	for(int i = 0; i < nov; i++) {
		xadj[i] = nnz;
		for(int j = 0; j < nov; j++) {
			assert(mat[(i * nov) + j] >= 0);
				if(mat[(i * nov) + j] > 0) {
					adj[nnz] = nov + j;
					val[nnz] = mat[(i * nov) + j];
					nnz++;
				}
		}  
	}

	for(int i = 0; i < nov; i++) {
		xadj[i + nov] = nnz;
		for(int j = 0; j < nov; j++) {
			assert(mat[(j * nov) + i] >= 0);
			if(mat[(j * nov) + i] > 0) {
				adj[nnz] = j;
				val[nnz] = mat[(j * nov) + i];
				nnz++;
			}
		}
	}
	
	xadj[2 * nov] = nnz;
}