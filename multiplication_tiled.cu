#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>

using std::cout;
using std::generate;
using std::vector;

__global__ void matrix_multiplication(const int* a, const int* b, int* c, int N, int tile_size){
    
    //shared memory
    __shared__ int s_a[tile_size * tile_size];
    __shared__ int s_b[tile_size * tile_size];

    //thread's row and col in 2d
    int row = blockIdx.y * tile_size + threadIdx.y;
    int col = blockIdx.x * tile_size + threadIdx.x;

    //tmp
    int tmp;

    for(int q = 0; q < N/tile_size; q++){
      // set shared memory
      s_a[threadIdx.y*tile_size + threadIdx.x] = a[row*N + q*tile_size + threadIdx.x];
      s_b[threadIdx.y*tile_size + threadIdx.x] = b[(q*tile_size+threadIdx.y)*N+col];
      
      //waiting all other threads finished setting
      __syncthreads();

      // calculate c in thread
      for(int k = 0; k < tile_size; k++){
        tmp += s_a[row * N + k] * s_b[k * N + col];
      }

      //waiting all other threads finished calculating
      __syncthreads();
    }

    //set calculated value to c matrix
    c[row * N + col] = tmp;
}

void verify_result(vector<int> &a, vector<int> &b, vector<int> &c) {
  // For every row...
  for (int i = 0; i < N; i++) {
    // For every column...
    for (int j = 0; j < N; j++) {
      // For every element in the row-column pair
      int tmp = 0;
      for (int k = 0; k < N; k++) {
        // Accumulate the partial results
        tmp += a[i * N + k] * b[k * N + j];
      }

      // Check against the CPU result
      assert(tmp == c[i * N + j]);
    }
  }
}

int main(){

  const int N = 1<<10;
  size_t bytes = sizeof(int) * N * N;

  //Host
  vector<int> h_a(N * N);
  vector<int> h_b(N * N);
  vector<int> h_c(N * N);

  generate(h_a.begin(), h_a.end(), []() { return rand() % 100; });
  generate(h_b.begin(), h_b.end(), []() { return rand() % 100; });

  //Device
  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

  // Thread 32, blcok 32
  int THREADS = 32;
  int BLOCKS = N / THREADS;

  dim3 threads(THREADS, THREADS);
  dim3 blocks(BLOCKS, BLOCKS);

  matrix_multiplication<<<blocks, threads>>>(d_a, d_b, d_c, N, THREADS);

  cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

  verify_result(h_a, h_b, h_c);

  cout << "COMPLETED SUCCESSFULLY\n";

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}