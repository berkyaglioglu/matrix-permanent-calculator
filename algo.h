#include <iostream>
#include <bitset>
#include <omp.h>
#include <string.h>
#include "util.h"
using namespace std;

template <class T>
unsigned long long int rasmussen_crs_ccs(T* mat, int nov, int number_of_times) {
  int total = 0;
  for (int i = 0; i < nov*nov; i++) {
    if (mat[i] > 0) {
      total++;
    }
  }

  int curr_elt_r = 0;
  int curr_elt_c = 0;
  int cptrs[nov + 1];
  int rows[total];
  int rptrs[nov + 1];
  int cols[total];
  
  for (int i = 0; i < nov; i++) {
    rptrs[i] = curr_elt_r;
    cptrs[i] = curr_elt_c;
    for(int j = 0; j < nov; j++) {
      if (mat[i*nov + j] > 0) {
        cols[curr_elt_r] = j;
        curr_elt_r++;        
      }
      if (mat[j*nov + i] > 0) {
        rows[curr_elt_c] = j;
        curr_elt_c++;
      }
    }
  }
  rptrs[nov] = curr_elt_r;
  cptrs[nov] = curr_elt_c;

  srand(time(0));

  int nt = omp_get_max_threads();

  double sum_perm = 0;
  double sum_zeros = 0;
    
  #pragma omp parallel for num_threads(nt) reduction(+:sum_perm) reduction(+:sum_zeros)
    for (int time = 0; time < number_of_times; time++) {
      int row_nnz[nov];
      bool col_extracted[nov];
      
      for (int i = 0; i < nov; i++) {
        row_nnz[i] = rptrs[i+1] - rptrs[i];
        col_extracted[i] = false;
      }
      
      double perm = 1;
      
      for (int row = 0; row < nov; row++) {
        // multiply permanent with number of nonzeros in the current row
        perm *= row_nnz[row];

        // choose the column to be extracted randomly
        int random = rand() % row_nnz[row];
        int col;
        for (int i = rptrs[row]; i < rptrs[row+1]; i++) {
          int c = cols[i];
          if (!col_extracted[c]) {
            if (random == 0) {
              col = c;
              break;
            } else {
              random--;
            }        
          }
        }

        // exract the column
        col_extracted[col] = true;

        // update number of nonzeros of the rows after extracting the column
        bool zero_row = false;
        for (int i = cptrs[col+1]-1; i >= cptrs[col]; i--) {
          int r = rows[i];
          if (r > row) {
            row_nnz[r]--;
            if (row_nnz[r] == 0) {
              zero_row = true;
              break;
            }
          } else {
            break;
          }
        }

        if (zero_row) {
          perm = 0;
          sum_zeros += 1;
          break;
        }
      }

      sum_perm += perm;
    }

  cout << "number of zeros: " << sum_zeros << endl;
  
  return (sum_perm / number_of_times);
}

template <class T>
unsigned long long int rasmussen(T* mat, int nov, int number_of_times) {
  T* mat_t = new T[nov * nov];
  
  for (int i = 0; i < nov; i++) {
    for (int j = 0; j < nov; j++) {
      mat_t[(j * nov) + i] = mat[(i * nov) + j];
    }
  }

  srand(time(0));

  int nt = omp_get_max_threads();

  double sum_perm = 0;
  double sum_zeros = 0;
    
  #pragma omp parallel for num_threads(nt) reduction(+:sum_perm) reduction(+:sum_zeros)
    for (int time = 0; time < number_of_times; time++) {
      int row_nnz[nov];
      bool col_extracted[nov];
      
      for (int i = 0; i < nov; i++) {
        row_nnz[i] = 0;
        col_extracted[i] = false;
        for (int j = 0; j < nov; j++) {
          if (mat[(i * nov) + j] != 0) {
            row_nnz[i] += 1;
          }
        }
      }
      
      double perm = 1;
      
      for (int row = 0; row < nov; row++) {
        // multiply permanent with number of nonzeros in the current row
        perm *= row_nnz[row];

        // choose the column to be extracted randomly
        int random = rand() % row_nnz[row];
        int col;
        for (int c = 0; c < nov; c++) {
          if (!col_extracted[c] && mat[row * nov + c] != 0) {
            if (random == 0) {
              col = c;
              break;
            } else {
              random--;
            }        
          }
        }

        // exract the column
        col_extracted[col] = true;

        // update number of nonzeros of the rows after extracting the column
        bool zero_row = false;
        for (int r = row + 1; r < nov; r++) {
          if (mat_t[col * nov + r] != 0) {
            row_nnz[r]--;
            if (row_nnz[r] == 0) {
              zero_row = true;
              break;
            }
          }
        }

        if (zero_row) {
          perm = 0;
          sum_zeros += 1;
          break;
        }
      }

      sum_perm += perm;
    }

  delete[] mat_t;

  cout << "number of zeros: " << sum_zeros << endl;
  
  return (sum_perm / number_of_times);
}


template <class T>
unsigned long long int approximation_perman64(T* mat, int nov) {
  double Xa = 1;
  
  T* A = new T[nov*nov];
  for (int i = 0; i < nov; i++) {
    for (int j = 0; j < nov; j++) {
      A[i*nov + j] = mat[i*nov + j];
    }
  }

  for (int i = 0; i < nov; i++) {
    // dulmage-mendehlson filter part
    T* A_filtered = new T[(nov - i) * (nov - i)];
    for (int index = 0; index < (nov - i)*(nov - i); index++) {
      A_filtered[index] = A[index];
    }

    int *xadj, *adj;
    T *val;
    matrix2graph(A_filtered, (nov - i), xadj, adj, val);

    dulmage_mendehlson(A_filtered, xadj, adj, (nov - i), (nov - i)*2);
    delete[] xadj;
    delete[] adj;
    delete[] val;

    int row = -1;
    for (int r = 0; r < (nov - i); r++) {
      for (int c = 0; c < (nov - i); c++) {
        if (A_filtered[r*(nov - i) + c] != 0) {
          row = r;
          break;
        }
      }
    }
    if (row == -1) {
      delete[] A;
      delete[] A_filtered;
      break;
    }

    // Scale part
    double* d_r = new double[nov - i];
    double* d_c = new double[nov - i];
    ScaleMatrix(A, (nov - i), d_r, d_c);
    
    // use scaled matrix for pj
    double sum_row_of_S = 0;
    double max_in_row_of_S = 0;
    int col;
    double s;
    for (int j = 0; j < nov - i; j++) {
      if (A[row*(nov - i) + j] != 0) {
        s = d_r[0] * A[row*(nov - i) + j] * d_c[j];
        sum_row_of_S += s;

        if (A_filtered[row*(nov - i) + j] != 0 && s > max_in_row_of_S) {
          max_in_row_of_S = s;
          col = j;
        }
      }
    }
    
    double pj = max_in_row_of_S / sum_row_of_S;
    Xa /= pj;


    // 

    T* A_prev = A;
    
    if (i != (nov - 1)) {
      int new_nov = nov - i - 1;
      
      A = new T[new_nov * new_nov];

      for (int r = 0; r < new_nov; r++) {
        for (int c = 0; c < new_nov; c++) {
          if (r < row && c < col) {
            A[r*new_nov + c] = A_prev[r*(new_nov + 1) + c];
          } else if (r < row) {
            A[r*new_nov + c] = A_prev[(r + 1)*(new_nov + 1) + c];
          } else if (c < col) {
            A[r*new_nov + c] = A_prev[r*(new_nov + 1) + (c + 1)];
          } else {
            A[r*new_nov + c] = A_prev[(r + 1)*(new_nov + 1) + (c + 1)];
          }
        }
      }
    }
    
    delete[] A_prev;
    delete[] A_filtered;
    delete[] d_r;
    delete[] d_c;

    // row col check if they are all zero, then break
    bool check = false;
    int row_elts = 0, col_elts = 0;
    for (int m = 0; m < (nov - i - 1); m++) {
      row_elts = 0; col_elts = 0;
      for (int n = 0; n < (nov - i - 1); n++) {
        if (A[m*(nov-i-1) + n] != 0) {
          row_elts++;
        }
        if (A[n*(nov-i-1) + m] != 0) {
          col_elts++;
        }
        if (row_elts != 0 && col_elts != 0) {
          break;
        }
      }
      if (row_elts == 0 || col_elts == 0) {
        check = true;
        break;
      }
    }

    if (check) {
      delete[] A;
      break;
    }
  }

  return Xa;
}

template <class T>
unsigned long long int parallel_perman64_with_ccs(T* mat, int nov, int* cptrs, int* rows, T* cvals) {
  const int a = omp_get_max_threads();
  double x[64];   
  double rs; //row sum
  double p = 1; //product of the elements in vector 'x'
  
  //create the x vector and initiate the permanent
  for (int j = 0; j < nov; j++) {
    rs = .0f;
    for (int k = 0; k < nov; k++) {
      rs += mat[(j * nov) + k];  // sum of row j
    }
    x[j] = mat[(j * nov) + (nov-1)] - rs/2;  // see Nijenhuis and Wilf - x vector entry
    p *= x[j];   // product of the elements in vector 'x'
  }

  unsigned long long one = 1;
  unsigned long long start = 1;
  unsigned long long end = (1ULL << (nov-1));
  
  int nt = omp_get_max_threads();
  unsigned long long chunk_size = end / nt + 1;

  #pragma omp parallel num_threads(nt) firstprivate(x)
  { 
    int tid = omp_get_thread_num();
    unsigned long long my_start = start + tid * chunk_size;
    unsigned long long my_end = min(start + ((tid+1) * chunk_size), end);
    
    double s;  //+1 or -1 
    double prod; //product of the elements in vector 'x'
    double my_p = 0;
    unsigned long long i = my_start;
    unsigned long long gray = (i-1) ^ ((i-1) >> 1);

    for (int k = 0; k < (nov-1); k++) {
      if ((gray >> k) & 1ULL) { // whether kth column should be added to x vector or not
        for (int j = cptrs[k]; j < cptrs[k+1]; j++) {
          x[rows[j]] += cvals[j]; // see Nijenhuis and Wilf - update x vector entries
        }
      }
    }

    prod = 1.0;
    int zero_num = 0;
    for (int j = 0; j < nov; j++) {
      if (x[j] == 0) {
        zero_num++;
      } else {
        prod *= x[j];  //product of the elements in vector 'x'
      }
    }
    int k;
    while (i < my_end) {
      //compute the gray code
      k = __builtin_ctzll(i);
      gray ^= (one << k); // Gray-code order: 1,3,2,6,7,5,4,12,13,15,...
      //decide if subtract of not - if the kth bit of gray is one then 1, otherwise -1
      s = ((one << k) & gray) ? 1 : -1;
      
      for (int j = cptrs[k]; j < cptrs[k+1]; j++) {
        if (x[rows[j]] == 0) {
          zero_num--;
          x[rows[j]] += s * cvals[j]; // see Nijenhuis and Wilf - update x vector entries
          prod *= x[rows[j]];  //product of the elements in vector 'x'
        } else {
          prod /= x[rows[j]];
          x[rows[j]] += s * cvals[j]; // see Nijenhuis and Wilf - update x vector entries
          if (x[rows[j]] == 0) {
            zero_num++;
          } else {
            prod *= x[rows[j]];  //product of the elements in vector 'x'
          }
        }
      }

      if(zero_num == 0) {
        my_p += ((i&1ULL)? -1.0:1.0) * prod; 
      }
      
      i++;
    }

    #pragma omp critical
      p += my_p;
  }

  return((4*(nov&1)-2) * p);
}

template <class T>
unsigned long long int parallel_perman64(T* mat, int nov) {
  const int a = omp_get_max_threads();
  double x[64];   
  double rs; //row sum
  double p = 1; //product of the elements in vector 'x'
  
  //create the x vector and initiate the permanent
  for (int j = 0; j < nov; j++) {
    rs = .0f;
    for (int k = 0; k < nov; k++) {
      rs += mat[(j * nov) + k];  // sum of row j
    }
    x[j] = mat[(j * nov) + (nov-1)] - rs/2;  // see Nijenhuis and Wilf - x vector entry
    p *= x[j];   // product of the elements in vector 'x'
  }

  //create the transpose of the matrix
  T* mat_t = new T[nov * nov];
  for (int i = 0; i < nov; i++) {
    for (int j = 0; j < nov; j++) {
      mat_t[(i * nov) + j] = mat[(j * nov) + i];
    }
  }

  unsigned long long one = 1;
  unsigned long long start = 1;
  unsigned long long end = (1ULL << (nov-1));
  
  int nt = omp_get_max_threads();
  unsigned long long chunk_size = end / nt + 1;

  #pragma omp parallel num_threads(nt) firstprivate(x)
  { 
    int tid = omp_get_thread_num();
    unsigned long long my_start = start + tid * chunk_size;
    unsigned long long my_end = min(start + ((tid+1) * chunk_size), end);
    
    double *xptr; 
    double s;  //+1 or -1 
    double prod; //product of the elements in vector 'x'
    double my_p = 0;
    unsigned long long i = my_start;
    unsigned long long gray = (i-1) ^ ((i-1) >> 1);

    for (int k = 0; k < (nov-1); k++) {
      if ((gray >> k) & 1ULL) { // whether kth column should be added to x vector or not
        xptr = (double*)x;
        for (int j = 0; j < nov; j++) {
          *xptr += mat_t[(k * nov) + j]; // see Nijenhuis and Wilf - update x vector entries
          xptr++;
        }
      }
    }
    
    int k;
    while (i < my_end) {
      //compute the gray code
      k = __builtin_ctzll(i);
      gray ^= (one << k); // Gray-code order: 1,3,2,6,7,5,4,12,13,15,...
      //decide if subtract of not - if the kth bit of gray is one then 1, otherwise -1
      s = ((one << k) & gray) ? 1 : -1;
      
      prod = 1.0;
      xptr = (double*)x;
      for (int j = 0; j < nov; j++) {
        *xptr += s * mat_t[(k * nov) + j]; // see Nijenhuis and Wilf - update x vector entries
        prod *= *xptr++;  //product of the elements in vector 'x'
      }

      my_p += ((i&1ULL)? -1.0:1.0) * prod; 
      i++;
    }

    #pragma omp critical
      p += my_p;
  }

  delete [] mat_t;

  return((4*(nov&1)-2) * p);
}


template <class T>
unsigned long long int perman64(T* mat, int nov) {
  double x[64];   
  double rs; //row sum
  double s;  //+1 or -1 
  double prod; //product of the elements in vector 'x'
  double p = 1; //product of the elements in vector 'x'
  double *xptr; 
  int j, k;
  unsigned long long i, tn11 = (1ULL << (nov-1)) - 1ULL;
  unsigned long long int gray;
  
  //create the x vector and initiate the permanent
  for (j = 0; j < nov; j++) {
    rs = .0f;
    for (k = 0; k < nov; k++)
      rs += mat[(j * nov) + k];  // sum of row j
    x[j] = mat[(j * nov) + (nov-1)] - rs/2;  // see Nijenhuis and Wilf - x vector entry
    p *= x[j];   // product of the elements in vector 'x'
  }

  //create the transpose of the matrix
  T* mat_t = new T[nov * nov];
  for (int i = 0; i < nov; i++) {
    for (int j = 0; j < nov; j++) {
      mat_t[(i * nov) + j] = mat[(j * nov) + i];
    }
  }

  gray = 0;
  unsigned long long one = 1;

  unsigned long long counter = 0;

  double t_start = omp_get_wtime();
  for (i = 1; i <= tn11; i++) {

    //compute the gray code
    k = __builtin_ctzll(i);
    gray ^= (one << k); // Gray-code order: 1,3,2,6,7,5,4,12,13,15,...
    //decide if subtract of not - if the kth bit of gray is one then 1, otherwise -1
    s = ((one << k) & gray) ? 1 : -1;
    
    counter++;
    prod = 1.0;
    xptr = (double*)x;
    for (j = 0; j < nov; j++) {
      *xptr += s * mat_t[(k * nov) + j]; // see Nijenhuis and Wilf - update x vector entries
      prod *= *xptr++;  //product of the elements in vector 'x'
    }

    p += ((i&1ULL)? -1.0:1.0) * prod; 
  }

  delete [] mat_t;

  return((4*(nov&1)-2) * p);
}


template <class T>
unsigned long long int brute_w(int *xadj, int *adj, T* val, int nov) {
  double perman = 0;
  double prod;

  int* matched = new int[nov];
  for(int i = 0; i < nov; i++) matched[i] = 0;
  
  int h_nov = nov/2;
  int* ptrs = new int[h_nov];
  for(int i = 0; i < h_nov; i++) {ptrs[i] = xadj[i];}

  matched[0] = adj[0];
  matched[adj[0]] = 1;
  ptrs[0] = 1;
  prod = val[0];

  int curr = 1;
  while(curr >= 0) {
    //clear the existing matching on current
    if(matched[curr] != 0) {
      prod /= val[ptrs[curr] - 1];
      matched[matched[curr]] = 0;
      matched[curr] = 0;
    }

    //check if we can increase the matching by matching curr
    int ptr = ptrs[curr];
    int partner;
    for(; ptr < xadj[curr + 1]; ptr++) {
      if(matched[adj[ptr]] == 0) {
  partner = adj[ptr];
  ptrs[curr] = ptr + 1;
  prod *= val[ptr];
  break;
      }
    }

    if(ptr < xadj[curr + 1]) { //we can extend matching
      if(curr == h_nov - 1) {
  perman += prod;
  prod /= val[ptr];   
  ptrs[curr] = xadj[curr];
  curr--;
      } else {
  matched[curr] = partner;
  matched[partner] = 1;
  curr++;
      }
    } else {
      ptrs[curr] = xadj[curr];
      curr--;
    }
  }
  return perman;
}

template <class T>
unsigned long long int sparse_perman64_w(int* xadj, int* adj, T* val, int nov) {
  double x[64];   
  double rs; //row sum
  double s;  //+1 or -1
  double prod; //product of the elements in vector 'x'
  double p = 0; //product of the elements in vector 'x'
  int j, k;
  int ptr;
  unsigned long long int i, tn11 = (1ULL << (nov-1)) - 1ULL;
  unsigned long long int gray;

  for (j = 0; j < nov; j++) {
    rs = .0f;
    for(int ptr = xadj[j]; ptr < xadj[j+1]; ptr++) {
      rs += val[ptr];
    }
    x[j] = -rs/(2.0f);
  }

  for(int ptr = xadj[2*nov - 1]; ptr < xadj[2*nov]; ptr++) {
    x[adj[ptr]] += val[ptr];
  }

  int nzeros = 0;
  prod = 1;
  for(j = 0; j < nov; j++) {
    if(x[j] == 0) {
      nzeros++;
    } else {
      prod *= x[j]; 
    }
  }
  
  if(nzeros == 0) {
    p = prod;
  } else {
    p = 0;
  }

  gray = 0;
  unsigned long long ctr = 0LL;
  unsigned long long one = 1;

  unsigned long long counter = 0;

  double t_start = omp_get_wtime();
  for (i = 1; i <= tn11; i++) {

    k = __builtin_ctzll(++ctr);
    gray ^= (one << k); // Gray-code order: 1,3,2,6,7,5,4,12,13,15,...
    s = ((one << k) & gray) ? 1 : -1;

    if(s == 1) {
      for (ptr = xadj[nov + k]; ptr < xadj[nov + k + 1]; ptr++) {
  int index = adj[ptr];
  double value = x[index];
  T adding = val[ptr];
  
  if(value == 0 || value == -adding) {
    if(value == 0) {
      nzeros--;
      x[index] = adding;
      prod *= adding;
    } else {
      nzeros++;
      prod /= (-adding);
      x[index] = 0;
    }
  } else {
    x[index] += adding;
    prod /= value;
    prod *= (value + adding);
  }
      }
    } else {
      for (ptr = xadj[nov + k]; ptr < xadj[nov + k + 1]; ptr++) {
        int index = adj[ptr];
        double value = x[index];
  int adding = val[ptr];

        if(value == 0 || value == adding) {
          if(value == 0) {
            nzeros--;
            x[index] = -adding;
            prod *= (-adding);
          } else {
            nzeros++;
      prod /= value;
            x[index] = 0;
          }
        } else {
          x[index] -= adding;
          prod /= value;
          prod *= (value - adding);
        }
      }
    }
    
    if(nzeros == 0) {
      p += ((i&1ULL)? -1.0:1.0) * prod; 
      counter++;
    }
  }
  
  return((4*(nov&1)-2) * p);
}

template <class T>
unsigned long long int sparser_perman64_w(int* xadj, int* adj, T* val, int nov) {
  double x[64];   
    
  double rs; //row sum
  double s;  //+1 or -1
  double prod; //product of the elements in vector 'x'
  double p = 0; //product of the elements in vector 'x'
  int j, k;
  int ptr;
  unsigned long long int i, tn11 = (1ULL << (nov-1)) - 1ULL;
  unsigned long long int gray;

  for (j = 0; j < nov; j++) {
    rs = .0f;
    for(int ptr = xadj[j]; ptr < xadj[j+1]; ptr++) {
      rs += val[ptr];
    }
    x[j] = -rs/(2.0f);
  }

  for(int ptr = xadj[2*nov - 1]; ptr < xadj[2*nov]; ptr++) {
    x[adj[ptr]] += val[ptr];
  }

  int nzeros = 0;
  prod = 1;
  for(j = 0; j < nov; j++) {
    if(x[j] == 0) {
      nzeros++;
    } else {
      prod *= x[j]; 
    }
  }

  if(nzeros == 0) {
    p = prod;
  } else {
    p = 0;
  }

  gray = 0;
  unsigned long long ctr = 0LL;
  unsigned long long one = 1;

  unsigned long long counter = 0;

  for (i = 1; i <= tn11; i++) {

    k = __builtin_ctzll(++ctr);
    gray ^= (one << k); // Gray-code order: 1,3,2,6,7,5,4,12,13,15,...
    s = ((one << k) & gray) ? 1 : -1;

    for (ptr = xadj[nov + k]; ptr < xadj[nov + k + 1]; ptr++) {
      int index = adj[ptr];
      if(x[index] == 0) 
        nzeros--;
      x[index] += s * val[ptr];
      if(x[index] == 0) 
        nzeros++;
    }

    if(nzeros == 0) {
      counter++;
      prod = 1;
      for(int j = 0; j < nov; j++) 
        prod *= x[j];
      p += ((i&1ULL)? -1.0:1.0) * prod; 
    }
  }

  return((4*(nov&1)-2) * p);
}

template <class T>
unsigned long long int sparser_skip_perman64_w(int* xadj, int* adj, T* val, int nov) {
  double p = 0, prod, s, rs, x[64]; //row sum
  int j, k, ptr, column_id;
  unsigned long long int i = 0, tn11 = (1ULL << (nov-1)) - 1ULL, gray, start, period, ci, steps, change_j, change_all, one = 1;

  //initialize the vector entries
  for (j = 0; j < nov; j++) {
    rs = .0f; for(int ptr = xadj[j]; ptr < xadj[j+1]; ptr++) rs += val[ptr];
    x[j] = -rs/(2.0f);
  }
  for(int ptr = xadj[2*nov - 1]; ptr < xadj[2*nov]; ptr++) {
    x[adj[ptr]] += val[ptr];
  }

  //find next i
  prod = 1;
  change_all = i + 1;
  for(int j = 0; j < nov; j++) {
    prod = prod * x[j];
    if(x[j] == 0) { 
      change_j = -1;
      //compute smallest i that may change this entry
      for (ptr = xadj[j]; ptr < xadj[j + 1]; ptr++) {
        //these are the columns that touches that row
        start = one << (adj[ptr] - nov); //start is the firt number of 0s
        period = start << 1; //this is the change period
        ci = start;
        if(i >= start) {
          steps = (i - start) / period;
          ci = start + ((steps + 1) * period); //this is next i value that changes this bit
        }
        if(ci < change_j) {
          change_j = ci;
        }
      }

      if(change_j > change_all) {
        change_all = change_j;
      }
    } 
  }
  i = change_all;
  p = prod;
  //cout << "t: " << x[31] << endl;
  unsigned long long prev_gray = 0;
  unsigned long long counter = 0;
  double t_start = omp_get_wtime();
  //  cout << omp_get_thread_num() << " " << i << " " << tn11 << endl;
  
  while (i <= tn11) {
    k = __builtin_ctzll(i+1);
    gray = i ^ (i >> 1);
    s = ((one << k) & gray) ? 1 : -1;

    unsigned long long gray_diff = prev_gray ^ gray;
    //cout << "X: " << gray_diff << endl;
    int l = 0;
    while(gray_diff > 0) { // this contains the bit to be updated
      unsigned long long onel = one << l;
      if(gray_diff & onel) { // if bit l is changed 
        gray_diff ^= onel;   // unset bit
        if(gray & onel) {    // do the update
          for (ptr = xadj[nov + l]; ptr < xadj[nov + l + 1]; ptr++) {
            x[adj[ptr]] += val[ptr];
          }
        }
        else {
          for (ptr = xadj[nov + l]; ptr < xadj[nov + l + 1]; ptr++) {
            x[adj[ptr]] -= val[ptr];
          }
        }
      }
      l++;
    }

    counter++;
    //cout << "T: " << x[31] << endl;
    prev_gray = gray;
    int last_zero = -1;
    prod = 1; 
    for(j = nov - 1; j >= 0; j--) {
      prod *= x[j];
      if(x[j] == 0) {
        last_zero = j;
        break;
      }
    }
    
    if(prod != 0) {
      p += ((i&1ULL)? -1.0:1.0) * prod; 
      i++;
    }
    else {
      change_j = -1;
      for (ptr = xadj[last_zero]; ptr < xadj[last_zero + 1]; ptr++) {
        start = one << adj[ptr] - nov; 
        period = start << 1; 
        ci = start;
        if(i >= start) {
          steps = (i - start) / period;
          ci = start + ((steps + 1) * period);
        }
        if(ci < change_j) {
          change_j = ci;
        }
      }
      i++;
      if(change_j > i) {
        i = change_j;
      } 
    }
  }
  p = ((4*(nov&1)-2) * p);
  return p;//((4*(nov&1)-2) * p);
}

template <class T>
unsigned long long int parallel_skip_perman64_w(int* xadj, int* adj, T* val, int* mat, int nov) {
  //first initialize the vector then we will copy it to ourselves
  double rs, x[64], p;
  int j, ptr, nt;
  unsigned long long ci, start, end, chunk_size, change_j;

  //initialize the vector entries                                                                                        
  for (j = 0; j < nov; j++) {
    rs = .0f; 
    for(ptr = xadj[j]; ptr < xadj[j+1]; ptr++) 
      rs += val[ptr];
    x[j] = -rs/(2.0f);
  }
  
  for(ptr = xadj[2*nov - 1]; ptr < xadj[2*nov]; ptr++) {
    x[adj[ptr]] += val[ptr];
  }

  //update perman with initial x
  double prod = 1;
  for(j = 0; j < nov; j++) {
    prod *= x[j];
  }
  p = prod;

  //find start location        
  start = 1;
  for(int j = 0; j < nov; j++) {
    if(x[j] == 0) {
      change_j = -1;
      for (ptr = xadj[j]; ptr < xadj[j + 1]; ptr++) {
        ci = 1ULL << (adj[ptr] - nov); 
        if(ci < change_j) {
          change_j = ci;
        }
      }
      if(change_j > start) {
        start = change_j;
      }
    }
  }

  end = (1ULL << (nov-1));
  
  nt = omp_get_max_threads();
  chunk_size = (end - start + 1) / nt + 1;

  #pragma omp parallel num_threads(nt) private(j, ptr, rs, ci, change_j) 
  {
    double my_x[64];
    memcpy(my_x, x, sizeof(double) * 64);
    
    int tid = omp_get_thread_num();
    unsigned long long my_start = start + tid * chunk_size;
    unsigned long long my_end = min(start + ((tid+1) * chunk_size), end);
    
    //update if neccessary
    double my_p = 0;

    unsigned long long my_gray;    
    unsigned long long my_prev_gray = 0;
    double s;
    int k, ptr, column_id, last_zero;
    unsigned long long period, steps, step_start;

    unsigned long long counter = 0;
    unsigned long long i = my_start;

    while (i < my_end) {
      k = __builtin_ctzll(i + 1);
      my_gray = i ^ (i >> 1);

      unsigned long long gray_diff = my_prev_gray ^ my_gray;
      //cout << "X: " << gray_diff << endl;
      j = 0;
      while(gray_diff > 0) { // this contains the bit to be updated
        unsigned long long onej = 1ULL << j;
        if(gray_diff & onej) { // if bit l is changed 
          gray_diff ^= onej;   // unset bit
          if(my_gray & onej) {    // do the update
            for (ptr = xadj[nov + j]; ptr < xadj[nov + j + 1]; ptr++) {
              my_x[adj[ptr]] += val[ptr];
            }
          }
          else {
            for (ptr = xadj[nov + j]; ptr < xadj[nov + j + 1]; ptr++) {
              my_x[adj[ptr]] -= val[ptr];
            }
          }
        }
        j++;
      }
      counter++;
      //cout << "T: " << my_x[31] << endl;
      my_prev_gray = my_gray;
      last_zero = -1;
      double my_prod = 1; 
      for(j = nov - 1; j >= 0; j--) {
        my_prod *= my_x[j];
        if(my_x[j] == 0) {
          last_zero = j;
          break;
        }
      }
      
      if(my_prod != 0) {
        my_p += ((i&1ULL)? -1.0:1.0) * my_prod;
        //  cout << "here contribution " << my_prod << " : i - " << i << endl;
        
        i++;
      }
      else {
        change_j = -1;
        for (ptr = xadj[last_zero]; ptr < xadj[last_zero + 1]; ptr++) {
          step_start = 1ULL << (adj[ptr] - nov); 
          period = step_start << 1; 
          ci = step_start;
          if(i >= step_start) {
            steps = (i - step_start) / period;
            ci = step_start + ((steps + 1) * period);
          }
          if(ci < change_j) {
            change_j = ci;
          }
        }
  
        i++;
        if(change_j > i) {
          i = change_j;
        } 
      }
    }

    #pragma omp critical
      p += my_p;
  }
  return ((4*(nov&1)-2) * p);
}

template <class T>
unsigned long long int parallel_skip_perman64_w_balanced(int* xadj, int* adj, T* val, int nov) {
  //first initialize the vector then we will copy it to ourselves
  double rs, x[64], p;
  int j, ptr, nt;
  unsigned long long ci, start, end, chunk_size, change_j;

  //initialize the vector entries                                                                                        
  for (j = 0; j < nov; j++) {
    rs = .0f; for(ptr = xadj[j]; ptr < xadj[j+1]; ptr++) rs += val[ptr];
    x[j] = -rs/(2.0f);
  }
  for(ptr = xadj[2*nov - 1]; ptr < xadj[2*nov]; ptr++) {
    x[adj[ptr]] += val[ptr];
  }

  //update perman with initial x
  double prod = 1;
  for(j = 0; j < nov; j++) {
    prod *= x[j];
  }
  p = prod;

  //find start location        
  start = 1;
  for(int j = 0; j < nov; j++) {
    if(x[j] == 0) {
      change_j = -1;
      for (ptr = xadj[j]; ptr < xadj[j + 1]; ptr++) {
        ci = 1ULL << (adj[ptr] - nov); 
        if(ci < change_j) {
          change_j = ci;
        }
      }
      if(change_j > start) {
        start = change_j;
      }
    }
  }

  end = (1ULL << (nov-1));
  nt = omp_get_max_threads();

  int no_chunks = 512;
  chunk_size = (end - start + 1) / no_chunks + 1;

  #pragma omp parallel num_threads(nt) private(j, ptr, rs, ci, change_j) 
  {
    double my_x[64];
    
    #pragma omp for schedule(dynamic, 1)
      for(int cid = 0; cid < no_chunks; cid++) {
      //    int tid = omp_get_thread_num();
        unsigned long long my_start = start + cid * chunk_size;
        unsigned long long my_end = min(start + ((cid+1) * chunk_size), end);
      
        //update if neccessary
        double my_p = 0;
        
        unsigned long long my_gray;    
        unsigned long long my_prev_gray = 0;
        memcpy(my_x, x, sizeof(double) * 64);

        double s;
        int k, ptr, column_id, last_zero;
        unsigned long long period, steps, step_start;
        
        unsigned long long counter = 0;
        unsigned long long i = my_start;
        
        //#pragma omp critical 
        //cout << omp_get_thread_num() << " " << cid << " " << my_start << " " << my_end << " " << chunk_size << endl;
        
        double t_start = omp_get_wtime();
        while (i < my_end) {
          k = __builtin_ctzll(i + 1);
          my_gray = i ^ (i >> 1);
          
          unsigned long long gray_diff = my_prev_gray ^ my_gray;
          //cout << "X: " << gray_diff << endl;
          j = 0;
          while(gray_diff > 0) { // this contains the bit to be updated
            unsigned long long onej = 1ULL << j;
            if(gray_diff & onej) { // if bit l is changed 
              gray_diff ^= onej;   // unset bit
              if(my_gray & onej) {    // do the update
                for (ptr = xadj[nov + j]; ptr < xadj[nov + j + 1]; ptr++) {
                  my_x[adj[ptr]] += val[ptr];
                }
              }
              else {
                for (ptr = xadj[nov + j]; ptr < xadj[nov + j + 1]; ptr++) {
                  my_x[adj[ptr]] -= val[ptr];
                }
              }
            }
            j++;
          }
          counter++;
          //cout << "T: " << my_x[31] << endl;
          my_prev_gray = my_gray;
          last_zero = -1;
          double my_prod = 1; 
          for(j = nov - 1; j >= 0; j--) {
            my_prod *= my_x[j];
            if(my_x[j] == 0) {
              last_zero = j;
              break;
            }
          }
  
          if(my_prod != 0) {
            my_p += ((i&1ULL)? -1.0:1.0) * my_prod;
            //  cout << "here contribution " << my_prod << " : i - " << i << endl;
            
            i++;
          } 
          else {
            change_j = -1;
            for (ptr = xadj[last_zero]; ptr < xadj[last_zero + 1]; ptr++) {
              step_start = 1ULL << (adj[ptr] - nov); 
              period = step_start << 1; 
              ci = step_start;
              if(i >= step_start) {
                steps = (i - step_start) / period;
                ci = step_start + ((steps + 1) * period);
              }
              if(ci < change_j) {
                change_j = ci;
              }
            } 
            i++;
            if(change_j > i) {
              i = change_j;
            } 
          }
        }
      
        #pragma omp critical
          p += my_p;
      }
  }
    
  return ((4*(nov&1)-2) * p);
}
