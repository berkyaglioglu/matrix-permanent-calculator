#include <omp.h>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
using namespace std;


template <class T>
__global__ void kernel_xlocal(T* mat_t, double* x, double* p, int nov) {
  float my_x[40];
  for (int k = 0; k < nov; k++) {
    my_x[k] = x[k];
  }
  
  unsigned long long number_of_threads = blockDim.x * gridDim.x;

  unsigned long long one = 1;
  unsigned long long start = 1;
  unsigned long long end = (1ULL << (nov-1));
  
  unsigned long long chunk_size = end / number_of_threads + 1;

  
  int tid = threadIdx.x + (blockIdx.x * blockDim.x);
  unsigned long long my_start = start + tid * chunk_size;
  unsigned long long my_end = min(start + ((tid+1) * chunk_size), end);
    
  float *xptr; 
  double s;  //+1 or -1 
  double prod; //product of the elements in vector 'x'
  double my_p = 0;
  unsigned long long i = my_start;
  unsigned long long gray = (i-1) ^ ((i-1) >> 1);

  for (int k = 0; k < (nov-1); k++) {
    if ((gray >> k) & 1ULL) { // whether kth column should be added to x vector or not
      xptr = (float*)my_x;
      for (int j = 0; j < nov; j++) {
        *xptr += mat_t[(k * nov) + j]; // see Nijenhuis and Wilf - update x vector entries
        xptr++;
      }
    }
  }
    
  unsigned long long gray_diff;
  int k;
  while (i < my_end) {
    gray_diff = (i ^ (i >> 1)) ^ gray;
    k = 0;
    for (unsigned long long j = gray_diff; j > 1; j /= 2) {
      k++;
    }
    gray ^= (one << k); // Gray-code order: 1,3,2,6,7,5,4,12,13,15,...
    //decide if subtract of not - if the kth bit of gray is one then 1, otherwise -1
    s = ((one << k) & gray) ? 1 : -1;
      
    prod = 1.0;
    xptr = (float*)my_x;
    for (int j = 0; j < nov; j++) {
      *xptr += s * mat_t[(k * nov) + j]; // see Nijenhuis and Wilf - update x vector entries
      prod *= *xptr++;  //product of the elements in vector 'x'
    }

    my_p += ((i&1ULL)? -1.0:1.0) * prod; 
    i++;
  }

  p[tid] = my_p;
  
}

template <class T>
__global__ void kernel_xshared(T* mat_t, double* x, double* p, int nov) {
  int tid = threadIdx.x + (blockIdx.x * blockDim.x);
  int thread_id = threadIdx.x;

  extern __shared__ float shared_mem[]; 
  float *my_x = shared_mem; // size = nov * BLOCK_SIZE

  for (int k = 0; k < nov; k++) {
    my_x[thread_id*nov + k] = x[k];
  }
  
  unsigned long long number_of_threads = blockDim.x * gridDim.x;

  unsigned long long one = 1;
  unsigned long long start = 1;
  unsigned long long end = (1ULL << (nov-1));
  
  unsigned long long chunk_size = end / number_of_threads + 1;

  unsigned long long my_start = start + tid * chunk_size;
  unsigned long long my_end = min(start + ((tid+1) * chunk_size), end);
     
  double s;  //+1 or -1 
  double prod; //product of the elements in vector 'x'
  double my_p = 0;
  unsigned long long i = my_start;
  unsigned long long gray = (i-1) ^ ((i-1) >> 1);

  for (int k = 0; k < (nov-1); k++) {
    if ((gray >> k) & 1ULL) { // whether kth column should be added to x vector or not
      for (int j = 0; j < nov; j++) {
        my_x[thread_id*nov + j] += mat_t[(k * nov) + j]; // see Nijenhuis and Wilf - update x vector entries
      }
    }
  }
    
  unsigned long long gray_diff;
  int k;
  while (i < my_end) {
    gray_diff = (i ^ (i >> 1)) ^ gray;
    k = 0;
    for (unsigned long long j = gray_diff; j > 1; j /= 2) {
      k++;
    }
    gray ^= (one << k); // Gray-code order: 1,3,2,6,7,5,4,12,13,15,...
    //decide if subtract of not - if the kth bit of gray is one then 1, otherwise -1
    s = ((one << k) & gray) ? 1 : -1;
      
    prod = 1.0;
    for (int j = 0; j < nov; j++) {
      my_x[thread_id*nov + j] += s * mat_t[(k * nov) + j]; // see Nijenhuis and Wilf - update x vector entries
      prod *= my_x[thread_id*nov + j];  //product of the elements in vector 'x'
    }

    my_p += ((i&1ULL)? -1.0:1.0) * prod; 
    i++;
  }

  p[tid] = my_p;
  
}

template <class T>
__global__ void kernel_xshared_coalescing(T* mat_t, double* x, double* p, int nov) {
  int tid = threadIdx.x + (blockIdx.x * blockDim.x);
  int thread_id = threadIdx.x;
  int block_dim = blockDim.x;

  extern __shared__ float shared_mem[]; 
  float *my_x = shared_mem; // size = nov * BLOCK_SIZE

  for (int k = 0; k < nov; k++) {
    my_x[block_dim*k + thread_id] = x[k];
  }
  
  unsigned long long number_of_threads = blockDim.x * gridDim.x;

  unsigned long long one = 1;
  unsigned long long start = 1;
  unsigned long long end = (1ULL << (nov-1));
  
  unsigned long long chunk_size = end / number_of_threads + 1;

  unsigned long long my_start = start + tid * chunk_size;
  unsigned long long my_end = min(start + ((tid+1) * chunk_size), end);
  
  double s;  //+1 or -1 
  double prod; //product of the elements in vector 'x'
  double my_p = 0;
  unsigned long long i = my_start;
  unsigned long long gray = (i-1) ^ ((i-1) >> 1);

  for (int k = 0; k < (nov-1); k++) {
    if ((gray >> k) & 1ULL) { // whether kth column should be added to x vector or not
      for (int j = 0; j < nov; j++) {
        my_x[block_dim*j + thread_id] += mat_t[(k * nov) + j]; // see Nijenhuis and Wilf - update x vector entries
      }
    }
  }
    
  unsigned long long gray_diff;
  int k;
  while (i < my_end) {
    gray_diff = (i ^ (i >> 1)) ^ gray;
    k = 0;
    for (unsigned long long j = gray_diff; j > 1; j /= 2) {
      k++;
    }
    gray ^= (one << k); // Gray-code order: 1,3,2,6,7,5,4,12,13,15,...
    //decide if subtract of not - if the kth bit of gray is one then 1, otherwise -1
    s = ((one << k) & gray) ? 1 : -1;
      
    prod = 1.0;
    for (int j = 0; j < nov; j++) {
      my_x[block_dim*j + thread_id] += s * mat_t[(k * nov) + j]; // see Nijenhuis and Wilf - update x vector entries
      prod *= my_x[block_dim*j + thread_id];  //product of the elements in vector 'x'
    }

    my_p += ((i&1ULL)? -1.0:1.0) * prod; 
    i++;
  }

  p[tid] = my_p;
  
}

template <class T>
__global__ void kernel_xshared_coalescing_mshared(T* mat_t, double* x, double* p, int nov) {
  int tid = threadIdx.x + (blockIdx.x * blockDim.x);
  int thread_id = threadIdx.x;
  int block_dim = blockDim.x;

  extern __shared__ float shared_mem[]; 
  float *my_x = shared_mem; // size = nov * BLOCK_SIZE
  T *shared_mat_t = (T*) &my_x[nov * block_dim]; // size = nov * nov

  for (int k = 0; k < nov; k++) {
    my_x[block_dim*k + thread_id] = x[k];
  }

  for (int k = 0; k < ((nov*nov)/block_dim + 1); k++) {
    if ((block_dim * k + thread_id) < (nov * nov))
    shared_mat_t[block_dim * k + thread_id] = mat_t[block_dim * k + thread_id];
  }

  __syncthreads();

  unsigned long long number_of_threads = blockDim.x * gridDim.x;

  unsigned long long one = 1;
  unsigned long long start = 1;
  unsigned long long end = (1ULL << (nov-1));
  
  unsigned long long chunk_size = end / number_of_threads + 1;

  unsigned long long my_start = start + tid * chunk_size;
  unsigned long long my_end = min(start + ((tid+1) * chunk_size), end);
  
  double s;  //+1 or -1 
  double prod; //product of the elements in vector 'x'
  double my_p = 0;
  unsigned long long i = my_start;
  unsigned long long gray = (i-1) ^ ((i-1) >> 1);

  for (int k = 0; k < (nov-1); k++) {
    if ((gray >> k) & 1ULL) { // whether kth column should be added to x vector or not
      for (int j = 0; j < nov; j++) {
        my_x[block_dim*j + thread_id] += shared_mat_t[(k * nov) + j]; // see Nijenhuis and Wilf - update x vector entries
      }
    }
  }
    
  unsigned long long gray_diff;
  int k;
  while (i < my_end) {
    gray_diff = (i ^ (i >> 1)) ^ gray;
    k = 0;
    for (unsigned long long j = gray_diff; j > 1; j /= 2) {
      k++;
    }
    gray ^= (one << k); // Gray-code order: 1,3,2,6,7,5,4,12,13,15,...
    //decide if subtract of not - if the kth bit of gray is one then 1, otherwise -1
    s = ((one << k) & gray) ? 1 : -1;
      
    prod = 1.0;
    for (int j = 0; j < nov; j++) {
      my_x[block_dim*j + thread_id] += s * shared_mat_t[(k * nov) + j]; // see Nijenhuis and Wilf - update x vector entries
      prod *= my_x[block_dim*j + thread_id];  //product of the elements in vector 'x'
    }

    my_p += ((i&1ULL)? -1.0:1.0) * prod; 
    i++;
  }

  p[tid] = my_p;
  
}

template <class T>
__global__ void kernel_xglobal_mshared(T* mat_t, double *x_orig, double* x, double* p, int nov) {
  int thread_id = threadIdx.x;
  int block_id = blockIdx.x;
  int block_dim = blockDim.x;
  int grid_dim = gridDim.x;
  int tid = thread_id + (block_id * block_dim);

  extern __shared__ float shared_mem[]; 
  T *shared_mat_t = (T*) shared_mem; // size = nov * nov

  for (int k = 0; k < nov; k++) {
    x[block_dim*nov*block_id + block_dim*k + thread_id] = x_orig[k];
  }

  for (int k = 0; k < ((nov*nov)/block_dim + 1); k++) {
    if ((block_dim * k + thread_id) < (nov * nov))
    shared_mat_t[block_dim * k + thread_id] = mat_t[block_dim * k + thread_id];
  }

  __syncthreads();

  unsigned long long number_of_threads = block_dim * grid_dim;

  unsigned long long one = 1;
  unsigned long long start = 1;
  unsigned long long end = (1ULL << (nov-1));
  
  unsigned long long chunk_size = end / number_of_threads + 1;

  unsigned long long my_start = start + tid * chunk_size;
  unsigned long long my_end = min(start + ((tid+1) * chunk_size), end);
  
  double s;  //+1 or -1 
  double prod; //product of the elements in vector 'x'
  double my_p = 0;
  unsigned long long i = my_start;
  unsigned long long gray = (i-1) ^ ((i-1) >> 1);

  for (int k = 0; k < (nov-1); k++) {
    if ((gray >> k) & 1ULL) { // whether kth column should be added to x vector or not
      for (int j = 0; j < nov; j++) {
        x[block_dim*nov*block_id + block_dim*j + thread_id] += shared_mat_t[(k * nov) + j]; // see Nijenhuis and Wilf - update x vector entries
      }
    }
  }
    
  unsigned long long gray_diff;
  int k;
  while (i < my_end) {
    gray_diff = (i ^ (i >> 1)) ^ gray;
    k = 0;
    for (unsigned long long j = gray_diff; j > 1; j /= 2) {
      k++;
    }
    gray ^= (one << k); // Gray-code order: 1,3,2,6,7,5,4,12,13,15,...
    //decide if subtract of not - if the kth bit of gray is one then 1, otherwise -1
    s = ((one << k) & gray) ? 1 : -1;
      
    prod = 1.0;
    for (int j = 0; j < nov; j++) {
      x[block_dim*nov*block_id + block_dim*j + thread_id] += s * shared_mat_t[(k * nov) + j]; // see Nijenhuis and Wilf - update x vector entries
      prod *= x[block_dim*nov*block_id + block_dim*j + thread_id];  //product of the elements in vector 'x'
    }

    my_p += ((i&1ULL)? -1.0:1.0) * prod; 
    i++;
  }

  p[tid] = my_p;
  
}

template <class T>
__global__ void kernel_rasmussen(T* mat, double* p, int nov) {
  int tid = threadIdx.x + (blockIdx.x * blockDim.x);
  int thread_id = threadIdx.x;
  int block_dim = blockDim.x;

  extern __shared__ float shared_mem[]; 
  T *shared_mat = (T*) shared_mem; // size = nov * nov

  for (int k = 0; k < ((nov*nov)/block_dim + 1); k++) {
    if ((block_dim * k + thread_id) < (nov * nov))
    shared_mat[block_dim * k + thread_id] = mat[block_dim * k + thread_id];
  }

  __syncthreads();

  curandState_t state;
  curand_init(tid,0,0,&state);

  long col_extracted = 0;
  
  double perm = 1;
  
  for (int row = 0; row < nov; row++) {
    // multiply permanent with number of nonzeros in the current row
    int nnz = 0;
    for (int c = 0; c < nov; c++) {
      if (!((col_extracted >> c) & 1L) && shared_mat[row * nov + c] != 0) {
        nnz++;
      }
    }
    if (nnz == 0) {
      perm = 0;
      break;
    }
    perm *= nnz;

    // choose the column to be extracted randomly
    int random = curand_uniform(&state) / (1.0 / float(nnz));
    int col;

    if (random >= nnz) {
      random = nnz - 1;
    }
    for (int c = 0; c < nov; c++) {
      if (!((col_extracted >> c) & 1L) && shared_mat[row * nov + c] != 0) {
        if (random == 0) {
          col = c;
          break;
        } else {
          random--;
        }        
      }
    }

    // exract the column
    col_extracted |= (1L << col);
  }

  p[tid] = perm;
}

template <class T>
__global__ void kernel_approximation(T* mat, double* p, float* d_r, float* d_c, int nov, int scale_intervals, int scale_times) {
  int tid = threadIdx.x + (blockIdx.x * blockDim.x);
  int thread_id = threadIdx.x;
  int block_dim = blockDim.x;

  extern __shared__ float shared_mem[]; 
  T *shared_mat = (T*) shared_mem; // size = nov * nov

  for (int k = 0; k < ((nov*nov)/block_dim + 1); k++) {
    if ((block_dim * k + thread_id) < (nov * nov))
    shared_mat[block_dim * k + thread_id] = mat[block_dim * k + thread_id];
  }

  __syncthreads();

  curandState_t state;
  curand_init(tid,0,0,&state);

  long col_extracted = 0;
  bool is_break;
  for (int i = 0; i < nov; i++) {
    d_r[tid*nov + i] = 1;
    d_c[tid*nov + i] = 1;
  }
  
  double perm = 1;
  
  for (int row = 0; row < nov; row++) {
    // Scale part
    if (row % scale_intervals == 0) {

      for (int k = 0; k < scale_times; k++) {

        for (int j = 0; j < nov; j++) {
          if (!((col_extracted >> j) & 1L)) {
            double col_sum = 0;
            for (int i = row; i < nov; i++) {
              col_sum += d_r[tid*nov + i] * shared_mat[i*nov + j];
            }
            if (col_sum == 0) {
              is_break = true;
              break;
            }
            d_c[tid*nov + j] = 1 / col_sum;
          }
        }
        if (is_break) {
          break;
        }

        for (int i = row; i < nov; i++) {
          double row_sum = 0;
          for (int j = 0; j < nov; j++) {
            if (!((col_extracted >> j) & 1L)) {
              row_sum += shared_mat[i*nov + j] * d_c[tid*nov + j];
            }
          }
          if (row_sum == 0) {
            is_break = true;
            break;
          }
          d_r[tid*nov + i] = 1 / row_sum;
        }
        if (is_break) {
          break;
        }
      }

    }

    if (is_break) {
      perm = 0;
      break;
    }

    // use scaled matrix for pj
    double sum_row_of_S = 0;
    for (int j = 0; j < nov; j++) {
      if (!((col_extracted >> j) & 1L) && shared_mat[(row * nov) + j] != 0) {
        sum_row_of_S += d_r[tid*nov + row] * d_c[tid*nov + j];
      }
    }
    if (sum_row_of_S == 0) {
      perm = 0;
      break;
    }

    double random = curand_uniform(&state) * sum_row_of_S;
    double temp = 0;
    double s, pj;
    int col;
    for (int j = 0; j < nov; j++) {
      if (!((col_extracted >> j) & 1L) && shared_mat[(row * nov) + j] != 0) {
        s = d_r[tid*nov + row] * d_c[tid*nov + j];
        temp += s;
        if (random <= temp) {
          col = j;
          pj = s / sum_row_of_S;
          break;
        }
      }
    }

    // update perm
    perm /= pj;

    // exract the column
    col_extracted |= (1L << col);
  }

  p[tid] = perm;
}

template <class T>
__global__ void kernel_approximation_shared_scale_vectors(T* mat, double* p, float* d_r, float* d_c, int nov, int scale_intervals, int scale_times) {
  int tid = threadIdx.x + (blockIdx.x * blockDim.x);
  int thread_id = threadIdx.x;
  int block_dim = blockDim.x;

  extern __shared__ float shared_mem[]; 
  T *shared_mat = (T*) shared_mem; // size = nov * nov
  float *shared_r = (float*) &shared_mat[nov * nov]; // size = nov
  float *shared_c = (float*) &shared_r[nov]; // size = nov

  for (int k = 0; k < ((nov*nov)/block_dim + 1); k++) {
    if ((block_dim * k + thread_id) < (nov * nov)) {
      shared_mat[block_dim * k + thread_id] = mat[block_dim * k + thread_id];
    }
    if ((block_dim * k + thread_id) < nov) {
      shared_r[block_dim * k + thread_id] = d_r[block_dim * k + thread_id];
      shared_c[block_dim * k + thread_id] = d_c[block_dim * k + thread_id];
    }
  }

  __syncthreads();

  curandState_t state;
  curand_init(tid,0,0,&state);

  long col_extracted = 0;
  
  double perm = 1;
  
  for (int row = 0; row < nov; row++) {
    // use scaled matrix for pj
    double sum_row_of_S = 0;
    for (int j = 0; j < nov; j++) {
      if (!((col_extracted >> j) & 1L) && shared_mat[(row * nov) + j] != 0) {
        sum_row_of_S += shared_r[row] * shared_c[j];
      }
    }
    if (sum_row_of_S == 0) {
      perm = 0;
      break;
    }

    double random = curand_uniform(&state) * sum_row_of_S;
    double temp = 0;
    double s, pj;
    int col;
    for (int j = 0; j < nov; j++) {
      if (!((col_extracted >> j) & 1L) && shared_mat[(row * nov) + j] != 0) {
        s = shared_r[row] * shared_c[j];
        temp += s;
        if (random <= temp) {
          col = j;
          pj = s / sum_row_of_S;
          break;
        }
      }
    }

    // update perm
    perm /= pj;

    // exract the column
    col_extracted |= (1L << col);
  }

  p[tid] = perm;
}




template <class T>
double gpu_perman64_xlocal(T* mat, int nov, int grid_dim, int block_dim) {
  double x[nov]; 
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

  cudaSetDevice(1);
  T *d_mat_t;
  double *d_x, *d_p;
  double *h_p = new double[grid_dim * block_dim];

  cudaMalloc( &d_x, (nov) * sizeof(double));
  cudaMalloc( &d_p, (grid_dim * block_dim) * sizeof(double));
  cudaMalloc( &d_mat_t, (nov * nov) * sizeof(T));

  cudaMemcpy( d_x, x, (nov) * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy( d_mat_t, mat_t, (nov * nov) * sizeof(T), cudaMemcpyHostToDevice);

  double stt = omp_get_wtime();
  kernel_xlocal<<< grid_dim , block_dim >>> (d_mat_t, d_x, d_p, nov);
  cudaDeviceSynchronize();
  double enn = omp_get_wtime();
  cout << "kernel" << " in " << (enn - stt) << endl;
  
  cudaMemcpy( h_p, d_p, grid_dim * block_dim * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_mat_t);
  cudaFree(d_x);
  cudaFree(d_p);

  for (int i = 0; i < grid_dim * block_dim; i++) {
    p += h_p[i];
  }

  delete [] mat_t;
  delete[] h_p;

  return((4*(nov&1)-2) * p);
}

template <class T>
double gpu_perman64_xshared(T* mat, int nov, int grid_dim, int block_dim) {
  double x[nov]; 
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

  cudaSetDevice(1);
  T *d_mat_t;
  double *d_x, *d_p;
  double *h_p = new double[grid_dim * block_dim];

  cudaMalloc( &d_x, (nov) * sizeof(double));
  cudaMalloc( &d_p, (grid_dim * block_dim) * sizeof(double));
  cudaMalloc( &d_mat_t, (nov * nov) * sizeof(T));

  cudaMemcpy( d_x, x, (nov) * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy( d_mat_t, mat_t, (nov * nov) * sizeof(T), cudaMemcpyHostToDevice);

  double stt = omp_get_wtime();
  kernel_xshared<<< grid_dim , block_dim , nov*block_dim*sizeof(float) >>> (d_mat_t, d_x, d_p, nov);
  cudaDeviceSynchronize();
  double enn = omp_get_wtime();
  cout << "kernel" << " in " << (enn - stt) << endl;
  
  cudaMemcpy( h_p, d_p, grid_dim * block_dim * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_mat_t);
  cudaFree(d_x);
  cudaFree(d_p);

  for (int i = 0; i < grid_dim * block_dim; i++) {
    p += h_p[i];
  }

  delete [] mat_t;
  delete[] h_p;

  return((4*(nov&1)-2) * p);
}

template <class T>
double gpu_perman64_xshared_coalescing(T* mat, int nov, int grid_dim, int block_dim) {
  double x[nov]; 
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

  cudaSetDevice(1);
  T *d_mat_t;
  double *d_x, *d_p;
  double *h_p = new double[grid_dim * block_dim];

  cudaMalloc( &d_x, (nov) * sizeof(double));
  cudaMalloc( &d_p, (grid_dim * block_dim) * sizeof(double));
  cudaMalloc( &d_mat_t, (nov * nov) * sizeof(T));

  cudaMemcpy( d_x, x, (nov) * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy( d_mat_t, mat_t, (nov * nov) * sizeof(T), cudaMemcpyHostToDevice);

  double stt = omp_get_wtime();
  kernel_xshared_coalescing<<< grid_dim , block_dim , nov*block_dim*sizeof(float) >>> (d_mat_t, d_x, d_p, nov);
  cudaDeviceSynchronize();
  double enn = omp_get_wtime();
  cout << "kernel" << " in " << (enn - stt) << endl;
  
  cudaMemcpy( h_p, d_p, grid_dim * block_dim * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_mat_t);
  cudaFree(d_x);
  cudaFree(d_p);

  for (int i = 0; i < grid_dim * block_dim; i++) {
    p += h_p[i];
  }

  delete [] mat_t;
  delete[] h_p;

  return((4*(nov&1)-2) * p);
}

template <class T>
double gpu_perman64_xshared_coalescing_mshared(T* mat, int nov, int grid_dim, int block_dim) {
  double x[nov]; 
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

  cudaSetDevice(1);
  T *d_mat_t;
  double *d_x, *d_p;
  double *h_p = new double[grid_dim * block_dim];

  cudaMalloc( &d_x, (nov) * sizeof(double));
  cudaMalloc( &d_p, (grid_dim * block_dim) * sizeof(double));
  cudaMalloc( &d_mat_t, (nov * nov) * sizeof(T));

  cudaMemcpy( d_x, x, (nov) * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy( d_mat_t, mat_t, (nov * nov) * sizeof(T), cudaMemcpyHostToDevice);

  double stt = omp_get_wtime();
  kernel_xshared_coalescing_mshared<<< grid_dim , block_dim , (nov*block_dim*sizeof(float) + nov*nov*sizeof(T)) >>> (d_mat_t, d_x, d_p, nov);
  cudaDeviceSynchronize();
  double enn = omp_get_wtime();
  cout << "kernel" << " in " << (enn - stt) << endl;
  
  cudaMemcpy( h_p, d_p, grid_dim * block_dim * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_mat_t);
  cudaFree(d_x);
  cudaFree(d_p);

  for (int i = 0; i < grid_dim * block_dim; i++) {
    p += h_p[i];
  }

  delete [] mat_t;
  delete[] h_p;

  return((4*(nov&1)-2) * p);
}

template <class T>
double gpu_perman64_xglobal(T* mat, int nov, int grid_dim, int block_dim) {
  double x[nov]; 
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

  cudaSetDevice(1);
  T *d_mat_t;
  double *d_x_orig, *d_x, *d_p;
  double *h_p = new double[grid_dim * block_dim];

  cudaMalloc( &d_x_orig, (nov) * sizeof(double));
  cudaMalloc( &d_x, (block_dim * grid_dim * nov) * sizeof(double));
  cudaMalloc( &d_p, (grid_dim * block_dim) * sizeof(double));
  cudaMalloc( &d_mat_t, (nov * nov) * sizeof(T));

  cudaMemcpy( d_x_orig, x, (nov) * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy( d_mat_t, mat_t, (nov * nov) * sizeof(T), cudaMemcpyHostToDevice);

  double stt = omp_get_wtime();
  kernel_xglobal_mshared<<< grid_dim , block_dim, nov*nov*sizeof(T) >>> (d_mat_t, d_x_orig, d_x, d_p, nov);
  cudaDeviceSynchronize();
  double enn = omp_get_wtime();
  cout << "kernel" << " in " << (enn - stt) << endl;
  
  cudaMemcpy( h_p, d_p, grid_dim * block_dim * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_mat_t);
  cudaFree(d_x_orig);
  cudaFree(d_x);
  cudaFree(d_p);

  for (int i = 0; i < grid_dim * block_dim; i++) {
    p += h_p[i];
  }

  delete [] mat_t;
  delete[] h_p;

  return((4*(nov&1)-2) * p);
}

template <class T>
double gpu_perman64_rasmussen(T* mat, int nov, int number_of_times) {
  int block_size = 1024;
  int grid_size = number_of_times / block_size + 1;

  cudaSetDevice(1);
  T *d_mat;
  double *d_p;
  double *h_p = new double[grid_size * block_size];

  cudaMalloc( &d_p, (grid_size * block_size) * sizeof(double));
  cudaMalloc( &d_mat, (nov * nov) * sizeof(T));

  cudaMemcpy( d_mat, mat, (nov * nov) * sizeof(T), cudaMemcpyHostToDevice);

  double stt = omp_get_wtime();
  kernel_rasmussen<<< grid_size , block_size , (nov*nov*sizeof(T)) >>> (d_mat, d_p, nov);
  cudaDeviceSynchronize();
  double enn = omp_get_wtime();
  cout << "kernel" << " in " << (enn - stt) << endl;
  
  cudaMemcpy( h_p, d_p, grid_size * block_size * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_mat);
  cudaFree(d_p);

  double p = 0;
  #pragma omp parallel for num_threads(omp_get_max_threads()) reduction(+:p)
    for (int i = 0; i < grid_size * block_size; i++) {
      p += h_p[i];
    }

  delete[] h_p;

  return (p / (grid_size * block_size));
}

template <class T>
double gpu_perman64_approximation(T* mat, int nov, int number_of_times, int scale_intervals, int scale_times) {
  int block_size = 1024;
  int grid_size = number_of_times / block_size + 1;

  cudaSetDevice(1);

  float *h_r, *h_c;
  double *h_p = new double[grid_size * block_size];

  T *d_mat;
  double *d_p;
  float *d_r, *d_c;

  cudaMalloc( &d_p, (grid_size * block_size) * sizeof(double));
  cudaMalloc( &d_mat, (nov * nov) * sizeof(T));

  cudaMemcpy( d_mat, mat, (nov * nov) * sizeof(T), cudaMemcpyHostToDevice);

  if (scale_intervals == -1) {
    h_r = new float[nov];
    h_c = new float[nov];
    for (int i = 0; i < nov; i++) {
      h_r[i] = 1;
      h_c[i] = 1;
    }
    cudaMalloc( &d_r, (nov) * sizeof(float));
    cudaMalloc( &d_c, (nov) * sizeof(float));

    #pragma omp parallel for num_threads(omp_get_max_threads())
      for (int k = 0; k < scale_times; k++) {
        for (int j = 0; j < nov; j++) {
          float col_sum = 0;
          for (int i = 0; i < nov; i++) {
            col_sum += h_r[i] * mat[i*nov + j];
          }
          h_c[j] = 1 / col_sum;
        }
        for (int i = 0; i < nov; i++) {
          float row_sum = 0;
          for (int j = 0; j < nov; j++) {
            row_sum += mat[i*nov + j] * h_c[j];
          }
          h_r[i] = 1 / row_sum;
        }
      }

    cudaMemcpy( d_r, h_r, (nov) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy( d_c, h_c, (nov) * sizeof(float), cudaMemcpyHostToDevice);

    double stt = omp_get_wtime();
    kernel_approximation_shared_scale_vectors<<< grid_size , block_size , (nov*nov*sizeof(T) + 2*nov*sizeof(float)) >>> (d_mat, d_p, d_r, d_c, nov, scale_intervals, scale_times);
    cudaDeviceSynchronize();
    double enn = omp_get_wtime();
    cout << "kernel" << " in " << (enn - stt) << endl;

  } else {
    cudaMalloc( &d_r, (nov * grid_size * block_size) * sizeof(float));
    cudaMalloc( &d_c, (nov * grid_size * block_size) * sizeof(float));

    double stt = omp_get_wtime();
    kernel_approximation<<< grid_size , block_size , (nov*nov*sizeof(T)) >>> (d_mat, d_p, d_r, d_c, nov, scale_intervals, scale_times);
    cudaDeviceSynchronize();
    double enn = omp_get_wtime();
    cout << "kernel" << " in " << (enn - stt) << endl;
  }
  
  cudaMemcpy( h_p, d_p, grid_size * block_size * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_mat);
  cudaFree(d_p);
  cudaFree(d_r);
  cudaFree(d_c);

  double p = 0;
  #pragma omp parallel for num_threads(omp_get_max_threads()) reduction(+:p)
    for (int i = 0; i < grid_size * block_size; i++) {
      p += h_p[i];
    }

  delete[] h_p;
  if (scale_intervals == -1) {
    delete[] h_r;
    delete[] h_c;
  }

  return (p / (grid_size * block_size));
}



// ####################################################################################################



template <class T>
__global__ void kernel_xlocal_sparse(int* cptrs, int* rows, T* cvals, double* x, double* p, int nov) {
  float my_x[40];
  for (int k = 0; k < nov; k++) {
    my_x[k] = x[k];
  }
  
  unsigned long long number_of_threads = blockDim.x * gridDim.x;

  unsigned long long one = 1;
  unsigned long long start = 1;
  unsigned long long end = (1ULL << (nov-1));
  
  unsigned long long chunk_size = end / number_of_threads + 1;

  
  int tid = threadIdx.x + (blockIdx.x * blockDim.x);
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
        my_x[rows[j]] += cvals[j]; // see Nijenhuis and Wilf - update x vector entries
      }
    }
  }

  prod = 1.0;
  int zero_num = 0;
  for (int j = 0; j < nov; j++) {
    if (my_x[j] == 0) {
      zero_num++;
    } else {
      prod *= my_x[j];  //product of the elements in vector 'x'
    }
  }
    
  unsigned long long gray_diff;
  int k;
  while (i < my_end) {
    gray_diff = (i ^ (i >> 1)) ^ gray;
    k = 0;
    for (unsigned long long j = gray_diff; j > 1; j /= 2) {
      k++;
    }
    gray ^= (one << k); // Gray-code order: 1,3,2,6,7,5,4,12,13,15,...
    //decide if subtract of not - if the kth bit of gray is one then 1, otherwise -1
    s = ((one << k) & gray) ? 1 : -1;
      
    for (int j = cptrs[k]; j < cptrs[k+1]; j++) {
      if (my_x[rows[j]] == 0) {
        zero_num--;
        my_x[rows[j]] += s * cvals[j]; // see Nijenhuis and Wilf - update x vector entries
        prod *= my_x[rows[j]];  //product of the elements in vector 'x'
      } else {
        prod /= my_x[rows[j]];
        my_x[rows[j]] += s * cvals[j]; // see Nijenhuis and Wilf - update x vector entries
        if (my_x[rows[j]] == 0) {
          zero_num++;
        } else {
          prod *= my_x[rows[j]];  //product of the elements in vector 'x'
        }
      }
    }

    if(zero_num == 0) {
      my_p += ((i&1ULL)? -1.0:1.0) * prod; 
    }
    i++;
  }

  p[tid] = my_p;
  
}

template <class T>
__global__ void kernel_xshared_sparse(int* cptrs, int* rows, T* cvals, double* x, double* p, int nov) {
  int tid = threadIdx.x + (blockIdx.x * blockDim.x);
  int thread_id = threadIdx.x;

  extern __shared__ float shared_mem[]; 
  float *my_x = shared_mem; // size = nov * BLOCK_SIZE

  for (int k = 0; k < nov; k++) {
    my_x[thread_id*nov + k] = x[k];
  }
  
  unsigned long long number_of_threads = blockDim.x * gridDim.x;

  unsigned long long one = 1;
  unsigned long long start = 1;
  unsigned long long end = (1ULL << (nov-1));
  
  unsigned long long chunk_size = end / number_of_threads + 1;

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
        my_x[thread_id*nov + rows[j]] += cvals[j]; // see Nijenhuis and Wilf - update x vector entries
      }
    }
  }

  prod = 1.0;
  int zero_num = 0;
  for (int j = 0; j < nov; j++) {
    if (my_x[thread_id*nov + j] == 0) {
      zero_num++;
    } else {
      prod *= my_x[thread_id*nov + j];  //product of the elements in vector 'x'
    }
  }
    
  unsigned long long gray_diff;
  int k;
  while (i < my_end) {
    gray_diff = (i ^ (i >> 1)) ^ gray;
    k = 0;
    for (unsigned long long j = gray_diff; j > 1; j /= 2) {
      k++;
    }
    gray ^= (one << k); // Gray-code order: 1,3,2,6,7,5,4,12,13,15,...
    //decide if subtract of not - if the kth bit of gray is one then 1, otherwise -1
    s = ((one << k) & gray) ? 1 : -1;

    for (int j = cptrs[k]; j < cptrs[k+1]; j++) {
      if (my_x[thread_id*nov + rows[j]] == 0) {
        zero_num--;
        my_x[thread_id*nov + rows[j]] += s * cvals[j]; // see Nijenhuis and Wilf - update x vector entries
        prod *= my_x[thread_id*nov + rows[j]];  //product of the elements in vector 'x'
      } else {
        prod /= my_x[thread_id*nov + rows[j]];
        my_x[thread_id*nov + rows[j]] += s * cvals[j]; // see Nijenhuis and Wilf - update x vector entries
        if (my_x[thread_id*nov + rows[j]] == 0) {
          zero_num++;
        } else {
          prod *= my_x[thread_id*nov + rows[j]];  //product of the elements in vector 'x'
        }
      }
    }

    if(zero_num == 0) {
      my_p += ((i&1ULL)? -1.0:1.0) * prod; 
    }
    i++;
  }

  p[tid] = my_p;
  
}

template <class T>
__global__ void kernel_xshared_coalescing_sparse(int* cptrs, int* rows, T* cvals, double* x, double* p, int nov) {
  int tid = threadIdx.x + (blockIdx.x * blockDim.x);
  int thread_id = threadIdx.x;
  int block_dim = blockDim.x;

  extern __shared__ float shared_mem[]; 
  float *my_x = shared_mem; // size = nov * BLOCK_SIZE

  for (int k = 0; k < nov; k++) {
    my_x[block_dim*k + thread_id] = x[k];
  }
  
  unsigned long long number_of_threads = blockDim.x * gridDim.x;

  unsigned long long one = 1;
  unsigned long long start = 1;
  unsigned long long end = (1ULL << (nov-1));
  
  unsigned long long chunk_size = end / number_of_threads + 1;

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
        my_x[block_dim*rows[j] + thread_id] += cvals[j]; // see Nijenhuis and Wilf - update x vector entries
      }
    }
  }

  prod = 1.0;
  int zero_num = 0;
  for (int j = 0; j < nov; j++) {
    if (my_x[block_dim*j + thread_id] == 0) {
      zero_num++;
    } else {
      prod *= my_x[block_dim*j + thread_id];  //product of the elements in vector 'x'
    }
  }
    
  unsigned long long gray_diff;
  int k;
  while (i < my_end) {
    gray_diff = (i ^ (i >> 1)) ^ gray;
    k = 0;
    for (unsigned long long j = gray_diff; j > 1; j /= 2) {
      k++;
    }
    gray ^= (one << k); // Gray-code order: 1,3,2,6,7,5,4,12,13,15,...
    //decide if subtract of not - if the kth bit of gray is one then 1, otherwise -1
    s = ((one << k) & gray) ? 1 : -1;
      
    for (int j = cptrs[k]; j < cptrs[k+1]; j++) {
      if (my_x[block_dim*rows[j] + thread_id] == 0) {
        zero_num--;
        my_x[block_dim*rows[j] + thread_id] += s * cvals[j]; // see Nijenhuis and Wilf - update x vector entries
        prod *= my_x[block_dim*rows[j] + thread_id];  //product of the elements in vector 'x'
      } else {
        prod /= my_x[block_dim*rows[j] + thread_id];
        my_x[block_dim*rows[j] + thread_id] += s * cvals[j]; // see Nijenhuis and Wilf - update x vector entries
        if (my_x[block_dim*rows[j] + thread_id] == 0) {
          zero_num++;
        } else {
          prod *= my_x[block_dim*rows[j] + thread_id];  //product of the elements in vector 'x'
        }
      }
    }

    if(zero_num == 0) {
      my_p += ((i&1ULL)? -1.0:1.0) * prod; 
    }
    i++;
  }

  p[tid] = my_p;
  
}

template <class T>
__global__ void kernel_xshared_coalescing_mshared_sparse(int* cptrs, int* rows, T* cvals, double* x, double* p, int nov, int total) {
  int tid = threadIdx.x + (blockIdx.x * blockDim.x);
  int thread_id = threadIdx.x;
  int block_dim = blockDim.x;

  extern __shared__ float shared_mem[]; 
  float *my_x = shared_mem; // size = nov * BLOCK_SIZE
  int *shared_cptrs = (int*) &my_x[nov * block_dim]; // size = nov + 1
  int *shared_rows = (int*) &shared_cptrs[nov + 1]; // size = total num of elts
  T *shared_cvals = (T*) &shared_rows[total]; // size = total num of elts

  for (int k = 0; k < nov; k++) {
    my_x[block_dim*k + thread_id] = x[k];
    shared_cptrs[k] = cptrs[k];
  }
  shared_cptrs[nov] = cptrs[nov];
  
  for (int k = 0; k < total; k++) {
    shared_rows[k] = rows[k];
    shared_cvals[k] = cvals[k];
  }

  __syncthreads();

  unsigned long long number_of_threads = blockDim.x * gridDim.x;

  unsigned long long one = 1;
  unsigned long long start = 1;
  unsigned long long end = (1ULL << (nov-1));
  
  unsigned long long chunk_size = end / number_of_threads + 1;

  unsigned long long my_start = start + tid * chunk_size;
  unsigned long long my_end = min(start + ((tid+1) * chunk_size), end);
  
  double s;  //+1 or -1 
  double prod; //product of the elements in vector 'x'
  double my_p = 0;
  unsigned long long i = my_start;
  unsigned long long gray = (i-1) ^ ((i-1) >> 1);

  for (int k = 0; k < (nov-1); k++) {
    if ((gray >> k) & 1ULL) { // whether kth column should be added to x vector or not
      for (int j = shared_cptrs[k]; j < shared_cptrs[k+1]; j++) {
        my_x[block_dim*shared_rows[j] + thread_id] += shared_cvals[j]; // see Nijenhuis and Wilf - update x vector entries
      }
    }
  }

  prod = 1.0;
  int zero_num = 0;
  for (int j = 0; j < nov; j++) {
    if (my_x[block_dim*j + thread_id] == 0) {
      zero_num++;
    } else {
      prod *= my_x[block_dim*j + thread_id];  //product of the elements in vector 'x'
    }
  }
    
  unsigned long long gray_diff;
  int k;
  while (i < my_end) {
    gray_diff = (i ^ (i >> 1)) ^ gray;
    k = 0;
    for (unsigned long long j = gray_diff; j > 1; j /= 2) {
      k++;
    }
    gray ^= (one << k); // Gray-code order: 1,3,2,6,7,5,4,12,13,15,...
    //decide if subtract of not - if the kth bit of gray is one then 1, otherwise -1
    s = ((one << k) & gray) ? 1 : -1;
      
    for (int j = shared_cptrs[k]; j < shared_cptrs[k+1]; j++) {
      if (my_x[block_dim*shared_rows[j] + thread_id] == 0) {
        zero_num--;
        my_x[block_dim*shared_rows[j] + thread_id] += s * shared_cvals[j]; // see Nijenhuis and Wilf - update x vector entries
        prod *= my_x[block_dim*shared_rows[j] + thread_id];  //product of the elements in vector 'x'
      } else {
        prod /= my_x[block_dim*shared_rows[j] + thread_id];
        my_x[block_dim*shared_rows[j] + thread_id] += s * shared_cvals[j]; // see Nijenhuis and Wilf - update x vector entries
        if (my_x[block_dim*shared_rows[j] + thread_id] == 0) {
          zero_num++;
        } else {
          prod *= my_x[block_dim*shared_rows[j] + thread_id];  //product of the elements in vector 'x'
        }
      }
    }

    if(zero_num == 0) {
      my_p += ((i&1ULL)? -1.0:1.0) * prod; 
    }
    i++;
  }

  p[tid] = my_p;
  
}

__global__ void kernel_rasmussen_sparse(int* rptrs, int* cols, double* p, int nov, int nnz) {
  int tid = threadIdx.x + (blockIdx.x * blockDim.x);
  int thread_id = threadIdx.x;
  int block_dim = blockDim.x;

  extern __shared__ float shared_mem[]; 
  int *shared_rptrs = (int*) shared_mem; // size = nov + 1
  int *shared_cols = (int*) &shared_rptrs[nov + 1]; // size = nnz

  int max;
  if (nnz > nov) {
    max = nnz;
  } else {
    max = nov + 1;
  }

  for (int k = 0; k < (max / block_dim + 1); k++) {
    if ((block_dim * k + thread_id) < nnz) {
      shared_cols[block_dim * k + thread_id] = cols[block_dim * k + thread_id];
    }
    if ((block_dim * k + thread_id) < (nov + 1)) {
      shared_rptrs[block_dim * k + thread_id] = rptrs[block_dim * k + thread_id];
    }
  }

  __syncthreads();

  curandState_t state;
  curand_init(tid,0,0,&state);

  long col_extracted = 0;
  
  double perm = 1;
  
  for (int row = 0; row < nov; row++) {
    // multiply permanent with number of nonzeros in the current row
    nnz = 0;
    for (int i = shared_rptrs[row]; i < shared_rptrs[row+1]; i++) {
      int c = shared_cols[i];
      if (!((col_extracted >> c) & 1L)) {
        nnz++;
      }
    }
    if (nnz == 0) {
      perm = 0;
      break;
    }
    perm *= nnz;

    // choose the column to be extracted randomly
    int random = curand_uniform(&state) / (1.0 / float(nnz));
    int col;

    if (random >= nnz) {
      random = nnz - 1;
    }
    for (int i = shared_rptrs[row]; i < shared_rptrs[row+1]; i++) {
      int c = shared_cols[i];
      if (!((col_extracted >> c) & 1L)) {
        if (random == 0) {
          col = c;
          break;
        } else {
          random--;
        }        
      }
    }

    // exract the column
    col_extracted |= (1L << col);
  }

  p[tid] = perm;
}

__global__ void kernel_approximation_sparse(int* rptrs, int* cols, int* cptrs, int* rows, double* p, float* d_r, float* d_c, int nov, int nnz, int scale_intervals, int scale_times) {
  int tid = threadIdx.x + (blockIdx.x * blockDim.x);
  int thread_id = threadIdx.x;
  int block_dim = blockDim.x;

  extern __shared__ float shared_mem[]; 
  int *shared_rptrs = (int*) shared_mem; // size = nov + 1
  int *shared_cols = (int*) &shared_rptrs[nov + 1]; // size = nnz
  int *shared_cptrs = (int*) &shared_cols[nnz]; // size = nov + 1
  int *shared_rows = (int*) &shared_cptrs[nov + 1]; // size = nnz

  int max;
  if (nnz > nov) {
    max = nnz;
  } else {
    max = nov + 1;
  }

  for (int k = 0; k < (max / block_dim + 1); k++) {
    if ((block_dim * k + thread_id) < nnz) {
      shared_cols[block_dim * k + thread_id] = cols[block_dim * k + thread_id];
      shared_rows[block_dim * k + thread_id] = rows[block_dim * k + thread_id];
    }
    if ((block_dim * k + thread_id) < (nov + 1)) {
      shared_rptrs[block_dim * k + thread_id] = rptrs[block_dim * k + thread_id];
      shared_cptrs[block_dim * k + thread_id] = cptrs[block_dim * k + thread_id];
    }
  }

  __syncthreads();

  curandState_t state;
  curand_init(tid,0,0,&state);

  long col_extracted = 0;
  bool is_break;
  for (int i = 0; i < nov; i++) {
    d_r[tid*nov + i] = 1;
    d_c[tid*nov + i] = 1;
  }
  
  double perm = 1;
  
  for (int row = 0; row < nov; row++) {
    // Scale part
    if (row % scale_intervals == 0) {

      for (int k = 0; k < scale_times; k++) {

        for (int j = 0; j < nov; j++) {
          if (!((col_extracted >> j) & 1L)) {
            double col_sum = 0;
            int r;
            for (int i = shared_cptrs[j]; i < shared_cptrs[j+1]; i++) {
              r = shared_rows[i];
              col_sum += d_r[tid*nov + r];
            }
            if (col_sum == 0) {
              is_break = true;
              break;
            }
            d_c[tid*nov + j] = 1 / col_sum;
          }
        }
        if (is_break) {
          break;
        }

        for (int i = row; i < nov; i++) {
          double row_sum = 0;
          int c;
          for (int j = shared_rptrs[i]; j < shared_rptrs[i+1]; j++) {
            c = shared_cols[j];
            if (!((col_extracted >> c) & 1L)) {
              row_sum += d_c[tid*nov + c];
            }
          }
          if (row_sum == 0) {
            is_break = true;
            break;
          }
          d_r[tid*nov + i] = 1 / row_sum;
        }
        if (is_break) {
          break;
        }
      }

    }

    if (is_break) {
      perm = 0;
      break;
    }

    // use scaled matrix for pj
    double sum_row_of_S = 0;
    for (int i = shared_rptrs[row]; i < shared_rptrs[row+1]; i++) {
      int c = shared_cols[i];
      if (!((col_extracted >> c) & 1L)) {
        sum_row_of_S += d_r[tid*nov + row] * d_c[tid*nov + c];
      }
    }
    if (sum_row_of_S == 0) {
      perm = 0;
      break;
    }

    double random = curand_uniform(&state) * sum_row_of_S;
    double temp = 0;
    double s, pj;
    int col;
    for (int i = shared_rptrs[row]; i < shared_rptrs[row+1]; i++) {
      int c = shared_cols[i];
      if (!((col_extracted >> c) & 1L)) {
        s = d_r[tid*nov + row] * d_c[tid*nov + c];
        temp += s;
        if (random <= temp) {
          col = c;
          pj = s / sum_row_of_S;
          break;
        }
      }
    }

    // update perm
    perm /= pj;

    // exract the column
    col_extracted |= (1L << col);
  }

  p[tid] = perm;
}

__global__ void kernel_approximation_shared_scale_vectors_sparse(int* rptrs, int* cols, int* cptrs, int* rows, double* p, float* d_r, float* d_c, int nov, int nnz, int scale_intervals, int scale_times) {
  int tid = threadIdx.x + (blockIdx.x * blockDim.x);
  int thread_id = threadIdx.x;
  int block_dim = blockDim.x;

  extern __shared__ float shared_mem[]; 
  int *shared_rptrs = (int*) shared_mem; // size = nov + 1
  int *shared_cols = (int*) &shared_rptrs[nov + 1]; // size = nnz
  int *shared_cptrs = (int*) &shared_cols[nnz]; // size = nov + 1
  int *shared_rows = (int*) &shared_cptrs[nov + 1]; // size = nnz
  float *shared_r = (float*) &shared_rows[nnz]; // size = nov
  float *shared_c = (float*) &shared_r[nov]; // size = nov

  int max;
  if (nnz > nov) {
    max = nnz;
  } else {
    max = nov + 1;
  }

  for (int k = 0; k < (max / block_dim + 1); k++) {
    if ((block_dim * k + thread_id) < nnz) {
      shared_cols[block_dim * k + thread_id] = cols[block_dim * k + thread_id];
      shared_rows[block_dim * k + thread_id] = rows[block_dim * k + thread_id];
    }
    if ((block_dim * k + thread_id) < (nov + 1)) {
      shared_rptrs[block_dim * k + thread_id] = rptrs[block_dim * k + thread_id];
      shared_cptrs[block_dim * k + thread_id] = cptrs[block_dim * k + thread_id];
      if ((block_dim * k + thread_id) != nov) {
        shared_r[block_dim * k + thread_id] = d_r[block_dim * k + thread_id];
        shared_c[block_dim * k + thread_id] = d_c[block_dim * k + thread_id];
      }
    }
  }

  __syncthreads();

  curandState_t state;
  curand_init(tid,0,0,&state);

  long col_extracted = 0;
  
  double perm = 1;
  
  for (int row = 0; row < nov; row++) {
    // use scaled matrix for pj
    double sum_row_of_S = 0;
    for (int i = shared_rptrs[row]; i < shared_rptrs[row+1]; i++) {
      int c = shared_cols[i];
      if (!((col_extracted >> c) & 1L)) {
        sum_row_of_S += shared_r[row] * shared_c[c];
      }
    }
    if (sum_row_of_S == 0) {
      perm = 0;
      break;
    }

    double random = curand_uniform(&state) * sum_row_of_S;
    double temp = 0;
    double s, pj;
    int col;
    for (int i = shared_rptrs[row]; i < shared_rptrs[row+1]; i++) {
      int c = shared_cols[i];
      if (!((col_extracted >> c) & 1L)) {
        s = shared_r[row] * shared_c[c];
        temp += s;
        if (random <= temp) {
          col = c;
          pj = s / sum_row_of_S;
          break;
        }
      }
    }

    // update perm
    perm /= pj;

    // exract the column
    col_extracted |= (1L << col);
  }

  p[tid] = perm;
}



template <class T>
double gpu_perman64_xlocal_sparse(T* mat, int* cptrs, int* rows, T* cvals, int nov, int grid_dim, int block_dim) {
  double x[nov]; 
  double rs; //row sum
  double p = 1; //product of the elements in vector 'x'
  int total = 0;
  
  //create the x vector and initiate the permanent
  for (int j = 0; j < nov; j++) {
    rs = .0f;
    for (int k = 0; k < nov; k++) {
      if (mat[(j * nov) + k] != 0) {
        total++;
        rs += mat[(j * nov) + k];  // sum of row j
      }
    }
    x[j] = mat[(j * nov) + (nov-1)] - rs/2;  // see Nijenhuis and Wilf - x vector entry
    p *= x[j];   // product of the elements in vector 'x'
  }

  cudaSetDevice(1);
  T *d_cvals;
  int *d_cptrs, *d_rows;
  double *d_x, *d_p;
  double *h_p = new double[grid_dim * block_dim];

  cudaMalloc( &d_x, (nov) * sizeof(double));
  cudaMalloc( &d_p, (grid_dim * block_dim) * sizeof(double));
  cudaMalloc( &d_cptrs, (nov + 1) * sizeof(int));
  cudaMalloc( &d_rows, (total) * sizeof(int));
  cudaMalloc( &d_cvals, (total) * sizeof(T));

  cudaMemcpy( d_x, x, (nov) * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy( d_cptrs, cptrs, (nov + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy( d_rows, rows, (total) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy( d_cvals, cvals, (total) * sizeof(T), cudaMemcpyHostToDevice);

  double stt = omp_get_wtime();
  kernel_xlocal_sparse<<< grid_dim , block_dim >>> (d_cptrs, d_rows, d_cvals, d_x, d_p, nov);
  cudaDeviceSynchronize();
  double enn = omp_get_wtime();
  cout << "kernel" << " in " << (enn - stt) << endl;
  
  cudaMemcpy( h_p, d_p, grid_dim * block_dim * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_x);
  cudaFree(d_p);
  cudaFree(d_cptrs);
  cudaFree(d_rows);
  cudaFree(d_cvals);

  for (int i = 0; i < grid_dim * block_dim; i++) {
    p += h_p[i];
  }

  delete[] h_p;

  return((4*(nov&1)-2) * p);
}

template <class T>
double gpu_perman64_xshared_sparse(T* mat, int* cptrs, int* rows, T* cvals, int nov, int grid_dim, int block_dim) {
  double x[nov]; 
  double rs; //row sum
  double p = 1; //product of the elements in vector 'x'
  int total = 0;
  
  //create the x vector and initiate the permanent
  for (int j = 0; j < nov; j++) {
    rs = .0f;
    for (int k = 0; k < nov; k++) {
      if (mat[(j * nov) + k] != 0) {
        total++;
        rs += mat[(j * nov) + k];  // sum of row j
      }
    }
    x[j] = mat[(j * nov) + (nov-1)] - rs/2;  // see Nijenhuis and Wilf - x vector entry
    p *= x[j];   // product of the elements in vector 'x'
  }

  cudaSetDevice(1);
  T *d_cvals;
  int *d_cptrs, *d_rows;
  double *d_x, *d_p;
  double *h_p = new double[grid_dim * block_dim];

  cudaMalloc( &d_x, (nov) * sizeof(double));
  cudaMalloc( &d_p, (grid_dim * block_dim) * sizeof(double));
  cudaMalloc( &d_cptrs, (nov + 1) * sizeof(int));
  cudaMalloc( &d_rows, (total) * sizeof(int));
  cudaMalloc( &d_cvals, (total) * sizeof(T));

  cudaMemcpy( d_x, x, (nov) * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy( d_cptrs, cptrs, (nov + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy( d_rows, rows, (total) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy( d_cvals, cvals, (total) * sizeof(T), cudaMemcpyHostToDevice);

  double stt = omp_get_wtime();
  kernel_xshared_sparse<<< grid_dim , block_dim , nov*block_dim*sizeof(float) >>> (d_cptrs, d_rows, d_cvals, d_x, d_p, nov);
  cudaDeviceSynchronize();
  double enn = omp_get_wtime();
  cout << "kernel" << " in " << (enn - stt) << endl;
  
  cudaMemcpy( h_p, d_p, grid_dim * block_dim * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_x);
  cudaFree(d_p);
  cudaFree(d_cptrs);
  cudaFree(d_rows);
  cudaFree(d_cvals);

  for (int i = 0; i < grid_dim * block_dim; i++) {
    p += h_p[i];
  }

  delete[] h_p;

  return((4*(nov&1)-2) * p);
}

template <class T>
double gpu_perman64_xshared_coalescing_sparse(T* mat, int* cptrs, int* rows, T* cvals, int nov, int grid_dim, int block_dim) {
  double x[nov]; 
  double rs; //row sum
  double p = 1; //product of the elements in vector 'x'
  int total = 0;
  
  //create the x vector and initiate the permanent
  for (int j = 0; j < nov; j++) {
    rs = .0f;
    for (int k = 0; k < nov; k++) {
      if (mat[(j * nov) + k] != 0) {
        total++;
        rs += mat[(j * nov) + k];  // sum of row j
      }
    }
    x[j] = mat[(j * nov) + (nov-1)] - rs/2;  // see Nijenhuis and Wilf - x vector entry
    p *= x[j];   // product of the elements in vector 'x'
  }

  cudaSetDevice(1);
  T *d_cvals;
  int *d_cptrs, *d_rows;
  double *d_x, *d_p;
  double *h_p = new double[grid_dim * block_dim];

  cudaMalloc( &d_x, (nov) * sizeof(double));
  cudaMalloc( &d_p, (grid_dim * block_dim) * sizeof(double));
  cudaMalloc( &d_cptrs, (nov + 1) * sizeof(int));
  cudaMalloc( &d_rows, (total) * sizeof(int));
  cudaMalloc( &d_cvals, (total) * sizeof(T));

  cudaMemcpy( d_x, x, (nov) * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy( d_cptrs, cptrs, (nov + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy( d_rows, rows, (total) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy( d_cvals, cvals, (total) * sizeof(T), cudaMemcpyHostToDevice);

  double stt = omp_get_wtime();
  kernel_xshared_coalescing_sparse<<< grid_dim , block_dim , nov*block_dim*sizeof(float) >>> (d_cptrs, d_rows, d_cvals, d_x, d_p, nov);
  cudaDeviceSynchronize();
  double enn = omp_get_wtime();
  cout << "kernel" << " in " << (enn - stt) << endl;
  
  cudaMemcpy( h_p, d_p, grid_dim * block_dim * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_x);
  cudaFree(d_p);
  cudaFree(d_cptrs);
  cudaFree(d_rows);
  cudaFree(d_cvals);

  for (int i = 0; i < grid_dim * block_dim; i++) {
    p += h_p[i];
  }

  delete[] h_p;

  return((4*(nov&1)-2) * p);
}

template <class T>
double gpu_perman64_xshared_coalescing_mshared_sparse(T* mat, int* cptrs, int* rows, T* cvals, int nov, int grid_dim, int block_dim) {
  double x[nov]; 
  double rs; //row sum
  double p = 1; //product of the elements in vector 'x'
  int total = 0;
  
  //create the x vector and initiate the permanent
  for (int j = 0; j < nov; j++) {
    rs = .0f;
    for (int k = 0; k < nov; k++) {
      if (mat[(j * nov) + k] != 0) {
        total++;
        rs += mat[(j * nov) + k];  // sum of row j
      }
    }
    x[j] = mat[(j * nov) + (nov-1)] - rs/2;  // see Nijenhuis and Wilf - x vector entry
    p *= x[j];   // product of the elements in vector 'x'
  }

  cudaSetDevice(1);
  T *d_cvals;
  int *d_cptrs, *d_rows;
  double *d_x, *d_p;
  double *h_p = new double[grid_dim * block_dim];

  cudaMalloc( &d_x, (nov) * sizeof(double));
  cudaMalloc( &d_p, (grid_dim * block_dim) * sizeof(double));
  cudaMalloc( &d_cptrs, (nov + 1) * sizeof(int));
  cudaMalloc( &d_rows, (total) * sizeof(int));
  cudaMalloc( &d_cvals, (total) * sizeof(T));

  cudaMemcpy( d_x, x, (nov) * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy( d_cptrs, cptrs, (nov + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy( d_rows, rows, (total) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy( d_cvals, cvals, (total) * sizeof(T), cudaMemcpyHostToDevice);

  double stt = omp_get_wtime();
  kernel_xshared_coalescing_mshared_sparse<<< grid_dim , block_dim , (nov*block_dim*sizeof(float) + (nov+1)*sizeof(int) + total*sizeof(int) + total*sizeof(T)) >>> (d_cptrs, d_rows, d_cvals, d_x, d_p, nov, total);
  cudaDeviceSynchronize();
  double enn = omp_get_wtime();
  cout << "kernel" << " in " << (enn - stt) << endl;
  
  cudaMemcpy( h_p, d_p, grid_dim * block_dim * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_x);
  cudaFree(d_p);
  cudaFree(d_cptrs);
  cudaFree(d_rows);
  cudaFree(d_cvals);

  for (int i = 0; i < grid_dim * block_dim; i++) {
    p += h_p[i];
  }

  delete[] h_p;

  return((4*(nov&1)-2) * p);
}

double gpu_perman64_rasmussen_sparse(int *rptrs, int *cols, int nov, int nnz, int number_of_times) {
  int block_size = 1024;
  int grid_size = number_of_times / block_size + 1;

  cudaSetDevice(1);

  double *h_p = new double[grid_size * block_size];

  int *d_rptrs, *d_cols;
  double *d_p;

  cudaMalloc( &d_rptrs, (nov + 1) * sizeof(int));
  cudaMalloc( &d_cols, (nnz) * sizeof(int));
  cudaMalloc( &d_p, (grid_size * block_size) * sizeof(double));

  cudaMemcpy( d_rptrs, rptrs, (nov + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy( d_cols, cols, (nnz) * sizeof(int), cudaMemcpyHostToDevice);

  double stt = omp_get_wtime();
  kernel_rasmussen_sparse<<< grid_size , block_size , ((nnz + nov + 1)*sizeof(int)) >>> (d_rptrs, d_cols, d_p, nov, nnz);
  cudaDeviceSynchronize();
  double enn = omp_get_wtime();
  cout << "kernel" << " in " << (enn - stt) << endl;
  
  cudaMemcpy( h_p, d_p, grid_size * block_size * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_rptrs);
  cudaFree(d_cols);
  cudaFree(d_p);

  double p = 0;
  #pragma omp parallel for num_threads(omp_get_max_threads()) reduction(+:p)
    for (int i = 0; i < grid_size * block_size; i++) {
      p += h_p[i];
    }

  delete[] h_p;

  return (p / (grid_size * block_size));
}

double gpu_perman64_approximation_sparse(int *cptrs, int *rows, int *rptrs, int *cols, int nov, int nnz, int number_of_times, int scale_intervals, int scale_times) {
  int block_size = 1024;
  int grid_size = number_of_times / block_size + 1;

  cudaSetDevice(1);

  float *h_r, *h_c;
  double *h_p = new double[grid_size * block_size];

  int *d_rptrs, *d_cols, *d_cptrs, *d_rows;
  float *d_r, *d_c;
  double *d_p;

  cudaMalloc( &d_rptrs, (nov + 1) * sizeof(int));
  cudaMalloc( &d_cols, (nnz) * sizeof(int));
  cudaMalloc( &d_cptrs, (nov + 1) * sizeof(int));
  cudaMalloc( &d_rows, (nnz) * sizeof(int));
  cudaMalloc( &d_p, (grid_size * block_size) * sizeof(double));

  cudaMemcpy( d_rptrs, rptrs, (nov + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy( d_cols, cols, (nnz) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy( d_cptrs, cptrs, (nov + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy( d_rows, rows, (nnz) * sizeof(int), cudaMemcpyHostToDevice);

  if (scale_intervals == -1) {
    h_r = new float[nov];
    h_c = new float[nov];
    for (int i = 0; i < nov; i++) {
      h_r[i] = 1;
      h_c[i] = 1;
    }
    cudaMalloc( &d_r, (nov) * sizeof(float));
    cudaMalloc( &d_c, (nov) * sizeof(float));

    #pragma omp parallel for num_threads(omp_get_max_threads())
      for (int k = 0; k < scale_times; k++) {
        for (int j = 0; j < nov; j++) {
          float col_sum = 0;
          int r;
          for (int i = cptrs[j]; i < cptrs[j+1]; i++) {
            r = rows[i];
            col_sum += h_r[r];
          }
          h_c[j] = 1 / col_sum;
        }
        for (int i = 0; i < nov; i++) {
          float row_sum = 0;
          int c;
          for (int j = rptrs[i]; j < rptrs[i+1]; j++) {
            c = cols[j];
            row_sum += h_c[c];
          }
          h_r[i] = 1 / row_sum;
        }
      }

    cudaMemcpy( d_r, h_r, (nov) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy( d_c, h_c, (nov) * sizeof(float), cudaMemcpyHostToDevice);

    double stt = omp_get_wtime();
    kernel_approximation_shared_scale_vectors_sparse<<< grid_size , block_size , (2*(nnz + nov + 1)*sizeof(int) + 2*nov*sizeof(float)) >>> (d_rptrs, d_cols, d_cptrs, d_rows, d_p, d_r, d_c, nov, nnz, scale_intervals, scale_times);
    cudaDeviceSynchronize();
    double enn = omp_get_wtime();
    cout << "kernel" << " in " << (enn - stt) << endl;

  } else {
    cudaMalloc( &d_r, (nov * grid_size * block_size) * sizeof(float));
    cudaMalloc( &d_c, (nov * grid_size * block_size) * sizeof(float));

    double stt = omp_get_wtime();
    kernel_approximation_sparse<<< grid_size , block_size , (2*(nnz + nov + 1)*sizeof(int)) >>> (d_rptrs, d_cols, d_cptrs, d_rows, d_p, d_r, d_c, nov, nnz, scale_intervals, scale_times);
    cudaDeviceSynchronize();
    double enn = omp_get_wtime();
    cout << "kernel" << " in " << (enn - stt) << endl;
  }
  
  cudaMemcpy( h_p, d_p, grid_size * block_size * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_rptrs);
  cudaFree(d_cols);
  cudaFree(d_cptrs);
  cudaFree(d_rows);
  cudaFree(d_r);
  cudaFree(d_c);
  cudaFree(d_p);

  double p = 0;
  #pragma omp parallel for num_threads(omp_get_max_threads()) reduction(+:p)
    for (int i = 0; i < grid_size * block_size; i++) {
      p += h_p[i];
    }

  delete[] h_p;
  if (scale_intervals == -1) {
    delete[] h_r;
    delete[] h_c;
  }

  return (p / (grid_size * block_size));
}