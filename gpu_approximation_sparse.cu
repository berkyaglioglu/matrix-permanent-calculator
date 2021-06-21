#include <omp.h>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include "util.h"
using namespace std;


double cpu_rasmussen_sparse(int *cptrs, int *rows, int *rptrs, int *cols, int nov, int random, int number_of_times, int threads) {

  srand(random);

  double sum_perm = 0;
  double sum_zeros = 0;
    
  #pragma omp parallel for num_threads(threads) reduction(+:sum_perm) reduction(+:sum_zeros)
    for (int time = 0; time < number_of_times; time++) {
      int row_nnz[nov];
      long col_extracted = 0;
      
      for (int i = 0; i < nov; i++) {
        row_nnz[i] = rptrs[i+1] - rptrs[i];
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
  
  return sum_perm;
}

double cpu_approximation_perman64_sparse(int *cptrs, int *rows, int *rptrs, int *cols, int nov, int random, int number_of_times, int scale_intervals, int scale_times, int threads) {

  srand(random);

  double sum_perm = 0;
  double sum_zeros = 0;
    
  #pragma omp parallel for num_threads(threads) reduction(+:sum_perm) reduction(+:sum_zeros)
    for (int time = 0; time < number_of_times; time++) {
      long col_extracted = 0;

      double Xa = 1;
      double d_r[nov];
      double d_c[nov];
      for (int i = 0; i < nov; i++) {
        d_r[i] = 1;
        d_c[i] = 1;
      }

      for (int row = 0; row < nov; row++) {
        // Scale part
        if ((scale_intervals != -1 || (scale_intervals == -1 && row == 0)) && row % scale_intervals == 0) {
          bool success = ScaleMatrix_sparse(cptrs, rows, rptrs, cols, nov, row, col_extracted, d_r, d_c, scale_times);
          if (!success) {
            Xa = 0;
            sum_zeros++;
            break;
          }
        }

        // use scaled matrix for pj
        double sum_row_of_S = 0;
        for (int i = rptrs[row]; i < rptrs[row+1]; i++) {
          int c = cols[i];
          if (!((col_extracted >> c) & 1L)) {
            sum_row_of_S += d_r[row] * d_c[c];
          }
        }
        if (sum_row_of_S == 0) {
          Xa = 0;
          sum_zeros++;
          break;
        }

        double random = (double(rand()) / RAND_MAX) * sum_row_of_S;
        double temp = 0;
        double s, pj;
        int col;
        for (int i = rptrs[row]; i < rptrs[row+1]; i++) {
          int c = cols[i];
          if (!((col_extracted >> c) & 1L)) {
            s = d_r[row] * d_c[c];
            temp += s;
            if (random <= temp) {
              col = c;
              pj = s / sum_row_of_S;
              break;
            }
          }
        }

        // update Xa
        Xa /= pj;
        
        // exract the column
        col_extracted |= (1L << col);

      }

      sum_perm += Xa;
    }
  
  return sum_perm;
}


__global__ void kernel_rasmussen_sparse(int* rptrs, int* cols, double* p, int nov, int nnz, int rand) {
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
  curand_init(rand*tid,0,0,&state);

  long col_extracted = 0;
  long row_extracted = 0;
  
  double perm = 1;
  int row;
  
  for (int k = 0; k < nov; k++) {
    // multiply permanent with number of nonzeros in the current row
    int min_nnz = nov+1;
    int nnz;
    for (int r = 0; r < nov; r++) {
      if (!((row_extracted >> r) & 1L)) {
        nnz = 0;
        for (int i = shared_rptrs[r]; i < shared_rptrs[r+1]; i++) {
          int c = shared_cols[i];
          if (!((col_extracted >> c) & 1L)) {
            nnz++;
          }
        }
        if (min_nnz > nnz) {
          min_nnz = nnz;
          row = r;
        }
      }
    }
    
    if (min_nnz == 0) {
      perm = 0;
      break;
    }
    perm *= min_nnz;

    // choose the column to be extracted randomly
    int random = curand_uniform(&state) / (1.0 / float(min_nnz));
    int col;

    if (random >= min_nnz) {
      random = min_nnz - 1;
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
    // exract the row
    row_extracted |= (1L << row);
  }

  p[tid] = perm;
}

__global__ void kernel_approximation_sparse(int* rptrs, int* cols, int* cptrs, int* rows, double* p, float* d_r, float* d_c, int nov, int nnz, int scale_intervals, int scale_times, int rand) {
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
  curand_init(rand*tid,0,0,&state);

  long col_extracted = 0;
  long row_extracted = 0;
  bool is_break;
  for (int i = 0; i < nov; i++) {
    d_r[tid*nov + i] = 1;
    d_c[tid*nov + i] = 1;
  }
  
  double perm = 1;
  double col_sum, row_sum;
  int row;
  int min;
  int nnz_curr;
  
  for (int iter = 0; iter < nov; iter++) {
    min=nov+1;
    for (int i = 0; i < nov; i++) {
      if (!((row_extracted >> i) & 1L)) {
        nnz_curr = 0;
        for (int j = shared_rptrs[i]; j < shared_rptrs[i+1]; j++) {
          int c = shared_cols[j];
          if (!((col_extracted >> c) & 1L)) {
            nnz_curr++;
          }
        }
        if (min > nnz_curr) {
          min = nnz_curr;
          row = i;
        }
      }
    }
    // Scale part
    if (row % scale_intervals == 0) {

      for (int k = 0; k < scale_times; k++) {

        for (int j = 0; j < nov; j++) {
          if (!((col_extracted >> j) & 1L)) {
            col_sum = 0;
            int r;
            for (int i = shared_cptrs[j]; i < shared_cptrs[j+1]; i++) {
              r = shared_rows[i];
              if (!((row_extracted >> r) & 1L)) {
                col_sum += d_r[tid*nov + r];
              }
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

        for (int i = 0; i < nov; i++) {
          if (!((row_extracted >> i) & 1L)) {
            row_sum = 0;
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
    // exract the row
    row_extracted |= (1L << row);
  }

  p[tid] = perm;
}


double gpu_perman64_rasmussen_sparse(int *rptrs, int *cols, int nov, int nnz, int number_of_times, bool grid_graph) {
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

  srand(time(0));
  double stt = omp_get_wtime();
  kernel_rasmussen_sparse<<< grid_size , block_size , ((nnz + nov + 1)*sizeof(int)) >>> (d_rptrs, d_cols, d_p, nov, nnz, rand());
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

double gpu_perman64_rasmussen_multigpucpu_chunks_sparse(int *cptrs, int *rows, int *rptrs, int *cols, int nov, int nnz, int number_of_times, int gpu_num, bool cpu, int threads, bool grid_graph) {
  int block_size = 1024;
  int grid_size = 1024;

  int cpu_chunk = 50000;
  int num_of_times_so_far = 0;

  double p = 0;
  double p_partial[gpu_num+1];
  double p_partial_times[gpu_num+1];
  for (int id = 0; id < gpu_num+1; id++) {
    p_partial[id] = 0;
    p_partial_times[id] = 0;
  }

  srand(time(0));

  omp_set_nested(1);
  omp_set_dynamic(0);
  #pragma omp parallel for num_threads(gpu_num+1)
    for (int id = 0; id < gpu_num+1; id++) {
      if (id == gpu_num) {
        if (cpu) {
          bool check = true;
          #pragma omp critical 
          {
            if (num_of_times_so_far < number_of_times) {
              num_of_times_so_far += cpu_chunk;
            } else {
              check = false;
            }
          }
          while (check) {
            double stt = omp_get_wtime();
            p_partial[id] += cpu_rasmussen_sparse(cptrs, rows, rptrs, cols, nov, rand(), number_of_times, threads);
            double enn = omp_get_wtime();
            p_partial_times[id] += cpu_chunk;
            cout << "cpu" << " in " << (enn - stt) << endl;
            #pragma omp critical 
            {
              if (num_of_times_so_far < number_of_times) {
                num_of_times_so_far += cpu_chunk;
              } else {
                check = false;
              }
            }
          }
        }
      } else {
        bool check = true;
        #pragma omp critical 
        {
          if (num_of_times_so_far < number_of_times) {
            num_of_times_so_far += grid_size * block_size;
          } else {
            check = false;
          }
        }
        cudaSetDevice(id);
        int *d_rptrs, *d_cols;
        double *d_p;
        double *h_p = new double[grid_size * block_size];

        cudaMalloc( &d_p, (grid_size * block_size) * sizeof(double));
        cudaMalloc( &d_rptrs, (nov + 1) * sizeof(int));
        cudaMalloc( &d_cols, (nnz) * sizeof(int));

        cudaMemcpy( d_rptrs, rptrs, (nov + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy( d_cols, cols, (nnz) * sizeof(int), cudaMemcpyHostToDevice);

        while (check) {
          double stt = omp_get_wtime();
          kernel_rasmussen_sparse<<< grid_size , block_size , ((nnz + nov + 1)*sizeof(int)) >>> (d_rptrs, d_cols, d_p, nov, nnz, rand());
          cudaDeviceSynchronize();
          double enn = omp_get_wtime();
          cout << "kernel" << id << " in " << (enn - stt) << endl;

          cudaMemcpy( h_p, d_p, grid_size * block_size * sizeof(double), cudaMemcpyDeviceToHost);

          for (int i = 0; i < grid_size * block_size; i++) {
            p_partial[id] += h_p[i];
          }
          p_partial_times[id] += (grid_size * block_size);
          #pragma omp critical 
          {
            if (num_of_times_so_far < number_of_times) {
              num_of_times_so_far += grid_size * block_size;
            } else {
              check = false;
            }
          }
        }

        cudaFree(d_rptrs);
        cudaFree(d_cols);
        cudaFree(d_p);
        delete[] h_p;
      }
    }

  for (int id = 0; id < gpu_num+1; id++) {
    p += p_partial[id];
  }
  double times = 0;
  for (int id = 0; id < gpu_num+1; id++) {
    times += p_partial_times[id];
  }

  return p / times;
}

double gpu_perman64_approximation_sparse(int *cptrs, int *rows, int *rptrs, int *cols, int nov, int nnz, int number_of_times, int scale_intervals, int scale_times, bool grid_graph) {
  int block_size = 1024;
  int grid_size = number_of_times / block_size + 1;

  cudaSetDevice(1);

  double *h_p = new double[grid_size * block_size];

  int *d_rptrs, *d_cols, *d_cptrs, *d_rows;
  float *d_r, *d_c;
  double *d_p;

  cudaMalloc( &d_rptrs, (nov + 1) * sizeof(int));
  cudaMalloc( &d_cols, (nnz) * sizeof(int));
  cudaMalloc( &d_p, (grid_size * block_size) * sizeof(double));

  cudaMemcpy( d_rptrs, rptrs, (nov + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy( d_cols, cols, (nnz) * sizeof(int), cudaMemcpyHostToDevice);

  cudaMalloc( &d_cptrs, (nov + 1) * sizeof(int));
  cudaMalloc( &d_rows, (nnz) * sizeof(int));
  cudaMemcpy( d_cptrs, cptrs, (nov + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy( d_rows, rows, (nnz) * sizeof(int), cudaMemcpyHostToDevice);
  
  srand(time(0));

  cudaMalloc( &d_r, (nov * grid_size * block_size) * sizeof(float));
  cudaMalloc( &d_c, (nov * grid_size * block_size) * sizeof(float));

  double stt = omp_get_wtime();
  kernel_approximation_sparse<<< grid_size , block_size , (2*(nnz + nov + 1)*sizeof(int)) >>> (d_rptrs, d_cols, d_cptrs, d_rows, d_p, d_r, d_c, nov, nnz, scale_intervals, scale_times, rand());
  cudaDeviceSynchronize();
  double enn = omp_get_wtime();
  cout << "kernel" << " in " << (enn - stt) << endl;
  
  cudaMemcpy( h_p, d_p, grid_size * block_size * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_rptrs);
  cudaFree(d_cols);
  cudaFree(d_cptrs);
  cudaFree(d_rows);
  cudaFree(d_r);
  cudaFree(d_c);
  cudaFree(d_p);

  double p = 0;
  for (int i = 0; i < grid_size * block_size; i++) {
    p += h_p[i];
  }

  delete[] h_p;

  return (p / (grid_size * block_size));
}

double gpu_perman64_approximation_multigpucpu_chunks_sparse(int *cptrs, int *rows, int *rptrs, int *cols, int nov, int nnz, int number_of_times, int gpu_num, bool cpu, int scale_intervals, int scale_times, int threads, bool grid_graph) {
  int block_size = 1024;
  int grid_size = 1024;

  int cpu_chunk = 50000;
  int num_of_times_so_far = 0;

  double p = 0;
  double p_partial[gpu_num+1];
  double p_partial_times[gpu_num+1];
  for (int id = 0; id < gpu_num+1; id++) {
    p_partial[id] = 0;
    p_partial_times[id] = 0;
  }

  srand(time(0));

  omp_set_nested(1);
  omp_set_dynamic(0);
  #pragma omp parallel for num_threads(gpu_num+1)
    for (int id = 0; id < gpu_num+1; id++) {
      if (id == gpu_num) {
        if (cpu) {
          bool check = true;
          #pragma omp critical 
          {
            if (num_of_times_so_far < number_of_times) {
              num_of_times_so_far += cpu_chunk;
            } else {
              check = false;
            }
          }
          while (check) {
            double stt = omp_get_wtime();
            cpu_approximation_perman64_sparse(cptrs, rows, rptrs, cols, nov, rand(), number_of_times, scale_intervals, scale_times, threads);
            double enn = omp_get_wtime();
            p_partial_times[id] += cpu_chunk;
            cout << "cpu" << " in " << (enn - stt) << endl;
            #pragma omp critical 
            {
              if (num_of_times_so_far < number_of_times) {
                num_of_times_so_far += cpu_chunk;
              } else {
                check = false;
              }
            }
          }
        }
      } else {
        bool check = true;
        #pragma omp critical 
        {
          if (num_of_times_so_far < number_of_times) {
            num_of_times_so_far += grid_size * block_size;
          } else {
            check = false;
          }
        }
        cudaSetDevice(id);

        float *h_r, *h_c;
        double *h_p = new double[grid_size * block_size];

        int *d_rptrs, *d_cols, *d_cptrs, *d_rows;
        double *d_p;
        float *d_r, *d_c;

        cudaMalloc( &d_p, (grid_size * block_size) * sizeof(double));
        cudaMalloc( &d_rptrs, (nov + 1) * sizeof(int));
        cudaMalloc( &d_cols, (nnz) * sizeof(int));

        cudaMemcpy( d_rptrs, rptrs, (nov + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy( d_cols, cols, (nnz) * sizeof(int), cudaMemcpyHostToDevice);

        cudaMalloc( &d_cptrs, (nov + 1) * sizeof(int));
        cudaMalloc( &d_rows, (nnz) * sizeof(int));
        cudaMemcpy( d_cptrs, cptrs, (nov + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy( d_rows, rows, (nnz) * sizeof(int), cudaMemcpyHostToDevice);

        cudaMalloc( &d_r, (nov * grid_size * block_size) * sizeof(float));
        cudaMalloc( &d_c, (nov * grid_size * block_size) * sizeof(float));

        while (check) {
          double stt = omp_get_wtime();
          kernel_approximation_sparse<<< grid_size , block_size , (2*(nnz + nov + 1)*sizeof(int)) >>> (d_rptrs, d_cols, d_cptrs, d_rows, d_p, d_r, d_c, nov, nnz, scale_intervals, scale_times, rand());
          cudaDeviceSynchronize();
          double enn = omp_get_wtime();
          cout << "kernel" << id << " in " << (enn - stt) << endl;

          cudaMemcpy( h_p, d_p, grid_size * block_size * sizeof(double), cudaMemcpyDeviceToHost);

          for (int i = 0; i < grid_size * block_size; i++) {
            p_partial[id] += h_p[i];
          }
          p_partial_times[id] += (grid_size * block_size);
          #pragma omp critical 
          {
            if (num_of_times_so_far < number_of_times) {
              num_of_times_so_far += grid_size * block_size;
            } else {
              check = false;
            }
          }
        }

        cudaFree(d_rptrs);
        cudaFree(d_cols);
        cudaFree(d_cptrs);
        cudaFree(d_rows);
        cudaFree(d_p);
        cudaFree(d_r);
        cudaFree(d_c);

        delete[] h_p;
      }
    }

  for (int id = 0; id < gpu_num+1; id++) {
    p += p_partial[id];
  }
  double times = 0;
  for (int id = 0; id < gpu_num+1; id++) {
    times += p_partial_times[id];
  }

  return p / times;
}