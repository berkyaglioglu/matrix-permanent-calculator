#include <omp.h>
#include <stdio.h>
using namespace std;

#define BLOCK_SIZE    256
#define GRID_SIZE    2048



template <class T>
__global__ void kernel_xlocal(T* mat_t, double* x, double* p, int nov) {
  float my_x[48];
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

  __shared__ float my_x[36 * BLOCK_SIZE];
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

  __shared__ float my_x[36 * BLOCK_SIZE];
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

  __shared__ float my_x[36 * BLOCK_SIZE];
  for (int k = 0; k < nov; k++) {
    my_x[block_dim*k + thread_id] = x[k];
  }

  __shared__ T shared_mat_t[36*36]; 
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
unsigned long long int gpu_perman64_xlocal(T* mat, int nov) {
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

  cudaSetDevice(1);
  T *d_mat_t;
  double *d_x, *d_p;
  double *h_p = new double[GRID_SIZE * BLOCK_SIZE];

  cudaMalloc( &d_x, (64) * sizeof(double));
  cudaMalloc( &d_p, (GRID_SIZE * BLOCK_SIZE) * sizeof(double));
  cudaMalloc( &d_mat_t, (nov * nov) * sizeof(int));

  cudaMemcpy( d_x, x, (64) * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy( d_mat_t, mat_t, (nov * nov) * sizeof(int), cudaMemcpyHostToDevice);

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

  cudaSetDevice(1);
  T *d_mat_t;
  double *d_x, *d_p;
  double *h_p = new double[GRID_SIZE * BLOCK_SIZE];

  cudaMalloc( &d_x, (64) * sizeof(double));
  cudaMalloc( &d_p, (GRID_SIZE * BLOCK_SIZE) * sizeof(double));
  cudaMalloc( &d_mat_t, (nov * nov) * sizeof(int));

  cudaMemcpy( d_x, x, (64) * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy( d_mat_t, mat_t, (nov * nov) * sizeof(int), cudaMemcpyHostToDevice);

  double stt = omp_get_wtime();
  kernel_xshared<<< GRID_SIZE , BLOCK_SIZE >>> (d_mat_t, d_x, d_p, nov);
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

  cudaSetDevice(1);
  T *d_mat_t;
  double *d_x, *d_p;
  double *h_p = new double[GRID_SIZE * BLOCK_SIZE];

  cudaMalloc( &d_x, (64) * sizeof(double));
  cudaMalloc( &d_p, (GRID_SIZE * BLOCK_SIZE) * sizeof(double));
  cudaMalloc( &d_mat_t, (nov * nov) * sizeof(int));

  cudaMemcpy( d_x, x, (64) * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy( d_mat_t, mat_t, (nov * nov) * sizeof(int), cudaMemcpyHostToDevice);

  double stt = omp_get_wtime();
  kernel_xshared_coalescing<<< GRID_SIZE , BLOCK_SIZE >>> (d_mat_t, d_x, d_p, nov);
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

  cudaSetDevice(1);
  T *d_mat_t;
  double *d_x, *d_p;
  double *h_p = new double[GRID_SIZE * BLOCK_SIZE];

  cudaMalloc( &d_x, (64) * sizeof(double));
  cudaMalloc( &d_p, (GRID_SIZE * BLOCK_SIZE) * sizeof(double));
  cudaMalloc( &d_mat_t, (nov * nov) * sizeof(int));

  cudaMemcpy( d_x, x, (64) * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy( d_mat_t, mat_t, (nov * nov) * sizeof(int), cudaMemcpyHostToDevice);

  double stt = omp_get_wtime();
  kernel_xshared_coalescing_mshared<<< GRID_SIZE , BLOCK_SIZE >>> (d_mat_t, d_x, d_p, nov);
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
