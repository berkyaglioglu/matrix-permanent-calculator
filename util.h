
#ifndef UTIL
#define UTIL

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <ctime>
#include <cassert>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
using namespace std;

#define etype int
#define vtype int

string seperator = "************************************************************************";


void match(int* col_ptrs, int* col_ids, int* match, int* row_match, int n, int m) {
  int* visited = (int*)malloc(sizeof(int) * m);
  int* stack = (int*)malloc(sizeof(int) * n);
  int* colptrs = (int*)malloc(sizeof(int) * n);
  int* lookahead = (int*)malloc(sizeof(int) * n);
  int* unmatched = (int*)malloc(sizeof(int) * n);

  int i, j, row, col, stack_col, temp, ptr, eptr, stack_last,
    stop = 0, pcount = 1, stack_end_ptr, nunmatched = 0, nextunmatched = 0,
    current_col, inc = 1;

  memset(visited, 0, sizeof(int) * m);
  memcpy(lookahead, col_ptrs, sizeof(int) * n);

  for(i = 0; i < n; i++) {
    if(match[i] == -1 && col_ptrs[i] != col_ptrs[i+1]) {
      unmatched[nunmatched++] = i;
    }
  }

  while(!stop) {
    stop = 1; stack_end_ptr = n;
    if(inc) {
      for(i = 0; i < nunmatched; i++) {
	current_col = unmatched[i];
	stack[0] = current_col; stack_last = 0; colptrs[current_col] = col_ptrs[current_col];

	while(stack_last > -1) {
	  stack_col = stack[stack_last];

	  eptr = col_ptrs[stack_col + 1];
	  for(ptr = lookahead[stack_col]; ptr < eptr && row_match[col_ids[ptr]] != -1; ptr++){}
	  lookahead[stack_col] = ptr + 1;

	  if(ptr >= eptr) {
	    for(ptr = colptrs[stack_col]; ptr < eptr; ptr++) {
	      temp = visited[col_ids[ptr]];
	      if(temp != pcount && temp != -1) {
		break;
	      }
	    }
	    colptrs[stack_col] = ptr + 1;

	    if(ptr == eptr) {
	      if(stop) {stack[--stack_end_ptr] = stack_col;}
	      --stack_last;
	      continue;
	    }

	    row = col_ids[ptr]; visited[row] = pcount;
	    col = row_match[row]; stack[++stack_last] = col; colptrs[col] = col_ptrs[col];
	  } else {
	    row = col_ids[ptr]; visited[row] = pcount;
	    while(row != -1){
	      col = stack[stack_last--];
	      temp = match[col];
	      match[col] = row; row_match[row] = col;
	      row = temp;
	    }
	    stop = 0;
	    break;
	  }
	}

	if(match[current_col] == -1) {
	  if(stop) {
	    for(j = stack_end_ptr + 1; j < n; j++) {
	      visited[match[stack[j]]] = -1;
	    }
	    stack_end_ptr = n;
	  } else {
	    unmatched[nextunmatched++] = current_col;
	  }
	}
      }
    } else {
      for(i = 0; i < nunmatched; i++) {
	current_col = unmatched[i];
	stack[0] = current_col; stack_last = 0; colptrs[current_col] = col_ptrs[current_col + 1] - 1;

	while(stack_last > -1) {
	  stack_col = stack[stack_last];

	  eptr = col_ptrs[stack_col + 1];
	  for(ptr = lookahead[stack_col]; ptr < eptr && row_match[col_ids[ptr]] != -1; ptr++){}
	  lookahead[stack_col] = ptr + 1;

	  if(ptr >= eptr) {
	    eptr = col_ptrs[stack_col] - 1;
	    for(ptr = colptrs[stack_col]; ptr > eptr; ptr--) {
	      temp = visited[col_ids[ptr]];
	      if(temp != pcount && temp != -1) {
		break;
	      }
	    }
	    colptrs[stack_col] = ptr - 1;

	    if(ptr == eptr) {
	      if(stop) {stack[--stack_end_ptr] = stack_col;}
	      --stack_last;
	      continue;
	    }

	    row = col_ids[ptr]; visited[row] = pcount;
	    col = row_match[row]; stack[++stack_last] = col;
	    colptrs[col] = col_ptrs[col + 1] - 1;

	  } else {
	    row = col_ids[ptr]; visited[row] = pcount;
	    while(row != -1){
	      col = stack[stack_last--];
	      temp = match[col];
	      match[col] = row; row_match[row] = col;
	      row = temp;
	    }
	    stop = 0;
	    break;
	  }
	}

	if(match[current_col] == -1) {
	  if(stop) {
	    for(j = stack_end_ptr + 1; j < n; j++) {
	      visited[match[stack[j]]] = -1;
	    }
	    stack_end_ptr = n;
	  } else {
	    unmatched[nextunmatched++] = current_col;
	  }
	}
      }
    }
    pcount++; nunmatched = nextunmatched; nextunmatched = 0; inc = !inc;
  }

  free(unmatched);
  free(lookahead);
  free(colptrs);
  free(stack);
  free(visited);
}


void reach(etype* xadj, vtype* adj, vtype nov, bool* visited, vtype* que, vtype source) {
  for(vtype i = 0; i < nov; i++) {
    visited[i] = false;
  }

  que[0] = source;
  visited[source] = true;
  vtype qp = 0, qe = 1;
  
  while(qp < qe) {
    vtype curr = que[qp++];
    
    for(etype ptr = xadj[curr]; ptr < xadj[curr + 1]; ptr++) {
      vtype nbr = adj[ptr];
      if(!visited[nbr]) {
      	visited[nbr] = true;
      	que[qe++] = nbr;
      }
    }
  }
}

template <class T>
void dulmage_mendehlson(T *mat, int *xadj, int *adj, int hnov, int nov) {
  int* rmatch = new int[hnov];
  int* cmatch = new int[hnov];
  vtype* nadj = new vtype[xadj[hnov]];
  for(etype ptr = 0; ptr < xadj[hnov]; ptr++) {nadj[ptr] = adj[ptr] - hnov;}
  for(vtype i = 0; i < hnov; i++) {rmatch[i] = cmatch[i] = -1;}
  match(xadj, nadj, rmatch, cmatch, hnov, hnov);
  
  vtype mcount = 0;
  for(vtype i = 0; i < hnov; i++) {
    if(rmatch[i] >= 0) {
      mcount++;
      if(cmatch[rmatch[i]] != i) {
      	cout << "Weird matching " << endl;
      	exit(1);
      }
    }
  }
  cout << "Match count is " << mcount << endl;
  if(mcount != hnov) {
    cout << "Perman is 0" << endl;
    exit(1);
  }

  vtype* gxadj = new vtype[hnov+1];
  etype* gadj = new etype[xadj[hnov]];
  
  gxadj[0] = 0;
  etype ptr = 0;
  for(vtype i = 0; i < hnov; i++) {
    vtype matched = rmatch[i];
    for(etype ptr2 = xadj[i]; ptr2 < xadj[i+1]; ptr2++) {
      vtype nbor = adj[ptr2];
      if(nbor != matched + hnov) {
	gadj[ptr++] = cmatch[nbor - hnov];
      }
    }
    gxadj[i+1] = ptr;
  }
  
  // printGraph(gxadj, gadj, val, hnov);

  int* component = new int[hnov];
  bool* visit1 = new bool[hnov];
  bool* visit2 = new bool[hnov];
  int* que = new int[hnov];
  for(vtype i = 0; i < hnov; i++) {
    component[i] = -1;
  }

  int cid = 0;
  for(vtype i = 0; i < hnov; i++) {
    if(component[i] == -1) {
      component[i] = cid;
      
      reach(gxadj, gadj, hnov, visit1, que, i);
      
      for(vtype j = 0; j < hnov; j++) {
	if(i != j && component[j] == -1 && visit1[j]) {
	  reach(gxadj, gadj, hnov, visit2, que, j);
	  
	  if(visit2[i]) {
	    component[j] = cid;
	  }
	}
      }
      cid++;
    }
  }
  
  cout << "comps: ";
  for(vtype i = 0; i < hnov; i++) {
    cout << component[i] << " ";
  }
  cout << endl;

  vtype erased = 0;
  ptr = 0;
  etype* xadj_t = new etype[nov+1];
  for(vtype i = 0; i <= nov; i++) {
    xadj_t[i] = xadj[i];
  }

  for(vtype i = 0; i < hnov; i++) {
    for(etype ptr2 = xadj_t[i]; ptr2 < xadj_t[i+1]; ptr2++) {
      if(component[i] == component[adj[ptr2] - hnov]) {
	adj[ptr++] = adj[ptr2];
      } else {
	mat[(i * hnov) + adj[ptr2] - hnov] = 0;
	erased++;
      }
    }
    xadj[i+1] = ptr;
  }

  
  for(vtype i = hnov; i < nov; i++) {
    for(etype ptr2 = xadj_t[i]; ptr2 < xadj_t[i+1]; ptr2++) {
      if(mat[adj[ptr2] * hnov + (i - hnov)] == 1) {
        adj[ptr++] = adj[ptr2];
      }
    }
    xadj[i+1] = ptr;
  }
  cout << "no erased edges: " << erased << endl;

  delete [] component;
  delete [] visit1;
  delete [] visit2;
  delete [] que;
  delete [] gxadj;
  delete [] gadj;
  delete [] nadj;
  delete [] rmatch;
  delete [] cmatch;
  delete [] xadj_t;
}


/*
void CreateMatrix(int dim, double density, bool binary, string file) {
	ofstream outFile(file);

	string text = "";
	int nnz = 0;

	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {
			if ((rand() % 100) < (100 * density)) {
				nnz++;
				if (binary) {
					text += to_string(i) + " " + to_string(j) + " 1\n";
				}
				else {
					text += to_string(i) + " " + to_string(j) + " " + to_string(rand() % 5 + 1) + "\n";
				}
			}
		}
	}

	outFile << dim << " " << nnz << endl;
	outFile << text;

	outFile.close();
}
*/

template <class T>
void ReadMatrix(T* & mat, ifstream & inFile, int nov, bool generic) {
	int i, j;
	T val;
	string line;

	while (getline(inFile, line)) {
		istringstream iss(line);
		if (!(iss >> i >> j >> val)) { continue; } // erroneous line
		if (generic) {
			mat[i * nov + j] = val;
		} else {
			mat[i * nov + j] = 1;
		}
	}
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


template <class T>
void matrix2compressed(T* mat, int*& cptrs, int*& rows, T*& cvals, int*& rptrs, int*& cols, T*& rvals, int nov, int nnz) {
	int curr_elt_r = 0;
	int curr_elt_c = 0;
	cptrs = new int[nov + 1];
	rows = new int[nnz];
	cvals = new T[nnz];
	rptrs = new int[nov + 1];
	cols = new int[nnz];
	rvals = new T[nnz];

	for (int i = 0; i < nov; i++) {
		rptrs[i] = curr_elt_r;
		cptrs[i] = curr_elt_c;
		for(int j = 0; j < nov; j++) {
			if (mat[i*nov + j] > 0) {
				cols[curr_elt_r] = j;
				rvals[curr_elt_r] = mat[i*nov + j];
				curr_elt_r++;        
			}
			if (mat[j*nov + i] > 0) {
				rows[curr_elt_c] = j;
				cvals[curr_elt_c] = mat[j*nov + i];
				curr_elt_c++;
			}
		}
	}
	rptrs[nov] = curr_elt_r;
	cptrs[nov] = curr_elt_c;
}


bool ScaleMatrix_sparse(int *cptrs, int *rows, int *rptrs, int *cols, int nov, int row, long col_extracted, double d_r[], double d_c[], int scale_times) {
	
	for (int k = 0; k < scale_times; k++) {

		for (int j = 0; j < nov; j++) {
			if (!((col_extracted >> j) & 1L)) {
				double col_sum = 0;
				int r;
				for (int i = cptrs[j+1]-1; i >= cptrs[j]; i--) {
					r = rows[i];
					if (r >= row) {
						col_sum += d_r[r];
					} else {
						break;
					}
				}
				if (col_sum == 0) {
					return false;
				}
				d_c[j] = 1 / col_sum;
			}
		}
		for (int i = row; i < nov; i++) {
			double row_sum = 0;
			int c;
			for (int j = rptrs[i]; j < rptrs[i+1]; j++) {
				c = cols[j];
				if (!((col_extracted >> c) & 1L)) {
					row_sum += d_c[c];
				}
			}
			if (row_sum == 0) {
				return false;
			}
			d_r[i] = 1 / row_sum;
		}
	}

	return true;
}

template <class T>
bool ScaleMatrix(T* M, int nov, int row, long col_extracted, double d_r[], double d_c[], int scale_times) {
	
	for (int k = 0; k < scale_times; k++) {

		for (int j = 0; j < nov; j++) {
			if (!((col_extracted >> j) & 1L)) {
				double col_sum = 0;
				for (int i = row; i < nov; i++) {
					col_sum += d_r[i] * M[i*nov + j];
				}
				if (col_sum == 0) {
					return false;
				}
				d_c[j] = 1 / col_sum;
			}
		}
		for (int i = row; i < nov; i++) {
			double row_sum = 0;
			for (int j = 0; j < nov; j++) {
				if (!((col_extracted >> j) & 1L)) {
					row_sum += M[i*nov + j] * d_c[j];
				}
			}
			if (row_sum == 0) {
				return false;
			}
			d_r[i] = 1 / row_sum;
		}
	}

	return true;
}


#endif