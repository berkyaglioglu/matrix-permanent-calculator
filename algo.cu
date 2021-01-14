#include <omp.h>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
using namespace std;

#define BLOCK_SIZE    256
#define GRID_SIZE    2048



template <class T>
__global__ void kernel_xlocal(T* mat_t, double* x, double* p, int nov) {
  float my_x[36];
  for (int k = 0; k < nov; k++) {
    my_x[k] = x[k];
  }
  
  unsigned long long number_of_threads = BLOCK_SIZE * GRID_SIZE;

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
  
  unsigned long long number_of_threads = BLOCK_SIZE * GRID_SIZE;

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
  
  unsigned long long number_of_threads = BLOCK_SIZE * GRID_SIZE;

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
  T *shared_mat_t = (T*) &my_x[nov * BLOCK_SIZE]; // size = nov * nov

  for (int k = 0; k < nov; k++) {
    my_x[block_dim*k + thread_id] = x[k];
  }

  for (int k = 0; k < ((nov*nov)/block_dim + 1); k++) {
    if ((block_dim * k + thread_id) < (nov * nov))
    shared_mat_t[block_dim * k + thread_id] = mat_t[block_dim * k + thread_id];
  }

  __syncthreads();

  unsigned long long number_of_threads = BLOCK_SIZE * GRID_SIZE;

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
__global__ void kernel_rasmussen(T* mat, double* p, unsigned int seed, int nov) {
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
unsigned long long int gpu_perman64_xlocal(T* mat, int nov) {
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
  double *h_p = new double[GRID_SIZE * BLOCK_SIZE];

  cudaMalloc( &d_x, (nov) * sizeof(double));
  cudaMalloc( &d_p, (GRID_SIZE * BLOCK_SIZE) * sizeof(double));
  cudaMalloc( &d_mat_t, (nov * nov) * sizeof(T));

  cudaMemcpy( d_x, x, (nov) * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy( d_mat_t, mat_t, (nov * nov) * sizeof(T), cudaMemcpyHostToDevice);

  double stt = omp_get_wtime();
  kernel_xlocal<<< GRID_SIZE , BLOCK_SIZE >>> (d_mat_t, d_x, d_p, nov);
  cudaDeviceSynchronize();
  double enn = omp_get_wtime();
  cout << "kernel" << " in " << (enn - stt) << endl;
  
  cudaMemcpy( h_p, d_p, GRID_SIZE * BLOCK_SIZE * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_mat_t);
  cudaFree(d_x);
  cudaFree(d_p);

  for (int i = 0; i < GRID_SIZE * BLOCK_SIZE; i++) {
    p += h_p[i];
  }

  delete [] mat_t;
  delete[] h_p;

  return((4*(nov&1)-2) * p);
}

template <class T>
unsigned long long int gpu_perman64_xshared(T* mat, int nov) {
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
  double *h_p = new double[GRID_SIZE * BLOCK_SIZE];

  cudaMalloc( &d_x, (nov) * sizeof(double));
  cudaMalloc( &d_p, (GRID_SIZE * BLOCK_SIZE) * sizeof(double));
  cudaMalloc( &d_mat_t, (nov * nov) * sizeof(T));

  cudaMemcpy( d_x, x, (nov) * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy( d_mat_t, mat_t, (nov * nov) * sizeof(T), cudaMemcpyHostToDevice);

  double stt = omp_get_wtime();
  kernel_xshared<<< GRID_SIZE , BLOCK_SIZE , nov*BLOCK_SIZE*sizeof(float) >>> (d_mat_t, d_x, d_p, nov);
  cudaDeviceSynchronize();
  double enn = omp_get_wtime();
  cout << "kernel" << " in " << (enn - stt) << endl;
  
  cudaMemcpy( h_p, d_p, GRID_SIZE * BLOCK_SIZE * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_mat_t);
  cudaFree(d_x);
  cudaFree(d_p);

  for (int i = 0; i < GRID_SIZE * BLOCK_SIZE; i++) {
    p += h_p[i];
  }

  delete [] mat_t;
  delete[] h_p;

  return((4*(nov&1)-2) * p);
}

template <class T>
unsigned long long int gpu_perman64_xshared_coalescing(T* mat, int nov) {
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
  double *h_p = new double[GRID_SIZE * BLOCK_SIZE];

  cudaMalloc( &d_x, (nov) * sizeof(double));
  cudaMalloc( &d_p, (GRID_SIZE * BLOCK_SIZE) * sizeof(double));
  cudaMalloc( &d_mat_t, (nov * nov) * sizeof(T));

  cudaMemcpy( d_x, x, (nov) * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy( d_mat_t, mat_t, (nov * nov) * sizeof(T), cudaMemcpyHostToDevice);

  double stt = omp_get_wtime();
  kernel_xshared_coalescing<<< GRID_SIZE , BLOCK_SIZE , nov*BLOCK_SIZE*sizeof(float) >>> (d_mat_t, d_x, d_p, nov);
  cudaDeviceSynchronize();
  double enn = omp_get_wtime();
  cout << "kernel" << " in " << (enn - stt) << endl;
  
  cudaMemcpy( h_p, d_p, GRID_SIZE * BLOCK_SIZE * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_mat_t);
  cudaFree(d_x);
  cudaFree(d_p);

  for (int i = 0; i < GRID_SIZE * BLOCK_SIZE; i++) {
    p += h_p[i];
  }

  delete [] mat_t;
  delete[] h_p;

  return((4*(nov&1)-2) * p);
}

template <class T>
unsigned long long int gpu_perman64_xshared_coalescing_mshared(T* mat, int nov) {
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
  double *h_p = new double[GRID_SIZE * BLOCK_SIZE];

  cudaMalloc( &d_x, (nov) * sizeof(double));
  cudaMalloc( &d_p, (GRID_SIZE * BLOCK_SIZE) * sizeof(double));
  cudaMalloc( &d_mat_t, (nov * nov) * sizeof(T));

  cudaMemcpy( d_x, x, (nov) * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy( d_mat_t, mat_t, (nov * nov) * sizeof(T), cudaMemcpyHostToDevice);

  double stt = omp_get_wtime();
  kernel_xshared_coalescing_mshared<<< GRID_SIZE , BLOCK_SIZE , (nov*BLOCK_SIZE*sizeof(float) + nov*nov*sizeof(T)) >>> (d_mat_t, d_x, d_p, nov);
  cudaDeviceSynchronize();
  double enn = omp_get_wtime();
  cout << "kernel" << " in " << (enn - stt) << endl;
  
  cudaMemcpy( h_p, d_p, GRID_SIZE * BLOCK_SIZE * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_mat_t);
  cudaFree(d_x);
  cudaFree(d_p);

  for (int i = 0; i < GRID_SIZE * BLOCK_SIZE; i++) {
    p += h_p[i];
  }

  delete [] mat_t;
  delete[] h_p;

  return((4*(nov&1)-2) * p);
}

template <class T>
unsigned long long int gpu_perman64_rasmussen(T* mat, int nov) {
  int grid_size = 1024*1024;
  int block_size = 1024;

  cudaSetDevice(1);
  T *d_mat;
  double *d_p;
  double *h_p = new double[grid_size * block_size];

  cudaMalloc( &d_p, (grid_size * block_size) * sizeof(double));
  cudaMalloc( &d_mat, (nov * nov) * sizeof(T));

  cudaMemcpy( d_mat, mat, (nov * nov) * sizeof(T), cudaMemcpyHostToDevice);

  double stt = omp_get_wtime();
  kernel_rasmussen<<< grid_size , block_size , (nov*nov*sizeof(T)) >>> (d_mat, d_p, time(NULL), nov);
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


// ####################################################################################################



template <class T>
__global__ void kernel_xlocal_with_ccs(int* cptrs, int* rows, T* cvals, double* x, double* p, int nov) {
  float my_x[36];
  for (int k = 0; k < nov; k++) {
    my_x[k] = x[k];
  }
  
  unsigned long long number_of_threads = BLOCK_SIZE * GRID_SIZE;

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
__global__ void kernel_xshared_with_ccs(int* cptrs, int* rows, T* cvals, double* x, double* p, int nov) {
  int tid = threadIdx.x + (blockIdx.x * blockDim.x);
  int thread_id = threadIdx.x;

  extern __shared__ float shared_mem[]; 
  float *my_x = shared_mem; // size = nov * BLOCK_SIZE

  for (int k = 0; k < nov; k++) {
    my_x[thread_id*nov + k] = x[k];
  }
  
  unsigned long long number_of_threads = BLOCK_SIZE * GRID_SIZE;

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
__global__ void kernel_xshared_coalescing_with_ccs(int* cptrs, int* rows, T* cvals, double* x, double* p, int nov) {
  int tid = threadIdx.x + (blockIdx.x * blockDim.x);
  int thread_id = threadIdx.x;
  int block_dim = blockDim.x;

  extern __shared__ float shared_mem[]; 
  float *my_x = shared_mem; // size = nov * BLOCK_SIZE

  for (int k = 0; k < nov; k++) {
    my_x[block_dim*k + thread_id] = x[k];
  }
  
  unsigned long long number_of_threads = BLOCK_SIZE * GRID_SIZE;

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
__global__ void kernel_xshared_coalescing_mshared_with_ccs(int* cptrs, int* rows, T* cvals, double* x, double* p, int nov, int total) {
  int tid = threadIdx.x + (blockIdx.x * blockDim.x);
  int thread_id = threadIdx.x;
  int block_dim = blockDim.x;

  extern __shared__ float shared_mem[]; 
  float *my_x = shared_mem; // size = nov * BLOCK_SIZE
  int *shared_cptrs = (int*) &my_x[nov * BLOCK_SIZE]; // size = nov + 1
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

  unsigned long long number_of_threads = BLOCK_SIZE * GRID_SIZE;

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



template <class T>
unsigned long long int gpu_perman64_xlocal_with_ccs(T* mat, int* cptrs, int* rows, T* cvals, int nov) {
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
  double *h_p = new double[GRID_SIZE * BLOCK_SIZE];

  cudaMalloc( &d_x, (nov) * sizeof(double));
  cudaMalloc( &d_p, (GRID_SIZE * BLOCK_SIZE) * sizeof(double));
  cudaMalloc( &d_cptrs, (nov + 1) * sizeof(int));
  cudaMalloc( &d_rows, (total) * sizeof(int));
  cudaMalloc( &d_cvals, (total) * sizeof(T));

  cudaMemcpy( d_x, x, (nov) * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy( d_cptrs, cptrs, (nov + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy( d_rows, rows, (total) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy( d_cvals, cvals, (total) * sizeof(T), cudaMemcpyHostToDevice);

  double stt = omp_get_wtime();
  kernel_xlocal_with_ccs<<< GRID_SIZE , BLOCK_SIZE >>> (d_cptrs, d_rows, d_cvals, d_x, d_p, nov);
  cudaDeviceSynchronize();
  double enn = omp_get_wtime();
  cout << "kernel" << " in " << (enn - stt) << endl;
  
  cudaMemcpy( h_p, d_p, GRID_SIZE * BLOCK_SIZE * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_x);
  cudaFree(d_p);
  cudaFree(d_cptrs);
  cudaFree(d_rows);
  cudaFree(d_cvals);

  for (int i = 0; i < GRID_SIZE * BLOCK_SIZE; i++) {
    p += h_p[i];
  }

  delete[] h_p;

  return((4*(nov&1)-2) * p);
}

template <class T>
unsigned long long int gpu_perman64_xshared_with_css(T* mat, int* cptrs, int* rows, T* cvals, int nov) {
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
  double *h_p = new double[GRID_SIZE * BLOCK_SIZE];

  cudaMalloc( &d_x, (nov) * sizeof(double));
  cudaMalloc( &d_p, (GRID_SIZE * BLOCK_SIZE) * sizeof(double));
  cudaMalloc( &d_cptrs, (nov + 1) * sizeof(int));
  cudaMalloc( &d_rows, (total) * sizeof(int));
  cudaMalloc( &d_cvals, (total) * sizeof(T));

  cudaMemcpy( d_x, x, (nov) * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy( d_cptrs, cptrs, (nov + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy( d_rows, rows, (total) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy( d_cvals, cvals, (total) * sizeof(T), cudaMemcpyHostToDevice);

  double stt = omp_get_wtime();
  kernel_xshared_with_ccs<<< GRID_SIZE , BLOCK_SIZE , nov*BLOCK_SIZE*sizeof(float) >>> (d_cptrs, d_rows, d_cvals, d_x, d_p, nov);
  cudaDeviceSynchronize();
  double enn = omp_get_wtime();
  cout << "kernel" << " in " << (enn - stt) << endl;
  
  cudaMemcpy( h_p, d_p, GRID_SIZE * BLOCK_SIZE * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_x);
  cudaFree(d_p);
  cudaFree(d_cptrs);
  cudaFree(d_rows);
  cudaFree(d_cvals);

  for (int i = 0; i < GRID_SIZE * BLOCK_SIZE; i++) {
    p += h_p[i];
  }

  delete[] h_p;

  return((4*(nov&1)-2) * p);
}

template <class T>
unsigned long long int gpu_perman64_xshared_coalescing_with_ccs(T* mat, int* cptrs, int* rows, T* cvals, int nov) {
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
  double *h_p = new double[GRID_SIZE * BLOCK_SIZE];

  cudaMalloc( &d_x, (nov) * sizeof(double));
  cudaMalloc( &d_p, (GRID_SIZE * BLOCK_SIZE) * sizeof(double));
  cudaMalloc( &d_cptrs, (nov + 1) * sizeof(int));
  cudaMalloc( &d_rows, (total) * sizeof(int));
  cudaMalloc( &d_cvals, (total) * sizeof(T));

  cudaMemcpy( d_x, x, (nov) * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy( d_cptrs, cptrs, (nov + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy( d_rows, rows, (total) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy( d_cvals, cvals, (total) * sizeof(T), cudaMemcpyHostToDevice);

  double stt = omp_get_wtime();
  kernel_xshared_coalescing_with_ccs<<< GRID_SIZE , BLOCK_SIZE , nov*BLOCK_SIZE*sizeof(float) >>> (d_cptrs, d_rows, d_cvals, d_x, d_p, nov);
  cudaDeviceSynchronize();
  double enn = omp_get_wtime();
  cout << "kernel" << " in " << (enn - stt) << endl;
  
  cudaMemcpy( h_p, d_p, GRID_SIZE * BLOCK_SIZE * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_x);
  cudaFree(d_p);
  cudaFree(d_cptrs);
  cudaFree(d_rows);
  cudaFree(d_cvals);

  for (int i = 0; i < GRID_SIZE * BLOCK_SIZE; i++) {
    p += h_p[i];
  }

  delete[] h_p;

  return((4*(nov&1)-2) * p);
}

template <class T>
unsigned long long int gpu_perman64_xshared_coalescing_mshared_with_ccs(T* mat, int* cptrs, int* rows, T* cvals, int nov) {
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
  double *h_p = new double[GRID_SIZE * BLOCK_SIZE];

  cudaMalloc( &d_x, (nov) * sizeof(double));
  cudaMalloc( &d_p, (GRID_SIZE * BLOCK_SIZE) * sizeof(double));
  cudaMalloc( &d_cptrs, (nov + 1) * sizeof(int));
  cudaMalloc( &d_rows, (total) * sizeof(int));
  cudaMalloc( &d_cvals, (total) * sizeof(T));

  cudaMemcpy( d_x, x, (nov) * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy( d_cptrs, cptrs, (nov + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy( d_rows, rows, (total) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy( d_cvals, cvals, (total) * sizeof(T), cudaMemcpyHostToDevice);

  double stt = omp_get_wtime();
  kernel_xshared_coalescing_mshared_with_ccs<<< GRID_SIZE , BLOCK_SIZE , (nov*BLOCK_SIZE*sizeof(float) + (nov+1)*sizeof(int) + total*sizeof(int) + total*sizeof(T)) >>> (d_cptrs, d_rows, d_cvals, d_x, d_p, nov, total);
  cudaDeviceSynchronize();
  double enn = omp_get_wtime();
  cout << "kernel" << " in " << (enn - stt) << endl;
  
  cudaMemcpy( h_p, d_p, GRID_SIZE * BLOCK_SIZE * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_x);
  cudaFree(d_p);
  cudaFree(d_cptrs);
  cudaFree(d_rows);
  cudaFree(d_cvals);

  for (int i = 0; i < GRID_SIZE * BLOCK_SIZE; i++) {
    p += h_p[i];
  }

  delete[] h_p;

  return((4*(nov&1)-2) * p);
}
