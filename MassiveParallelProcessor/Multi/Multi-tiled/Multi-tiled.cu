#include<cuda_runtime.h>
#include<device_launch_parameters.h>

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>
#include<math.h>

using std::cout;
using std::generate;
using std::vector;

#define SHMEM_SIZE 32*32
#define TILE_SIZE 32

const int N = 1<<10;

__global__ void matrix_multiplication(const int* a, const int* b, int* c, int N){
    
    //shared memory
    __shared__ int s_a[SHMEM_SIZE];
    __shared__ int s_b[SHMEM_SIZE];

    //thread's row and col in 2d
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    //tmp
    int tmp = 0;

    for(int q = 0; q < N/TILE_SIZE; q++){
      // set shared memory
      s_a[threadIdx.y*TILE_SIZE + threadIdx.x] = a[row*N + q*TILE_SIZE + threadIdx.x];
      s_b[threadIdx.y*TILE_SIZE + threadIdx.x] = b[(q*TILE_SIZE+threadIdx.y)*N+col];
      
      //waiting all other threads finished setting
      __syncthreads();

      // calculate c in thread
      for(int k = 0; k < TILE_SIZE; k++){
        tmp += s_a[threadIdx.y * TILE_SIZE + k] * s_b[k * TILE_SIZE + threadIdx.x];
      }

      //waiting all other threads finished calculating
      __syncthreads();
    }

    //set calculated value to c matrix
    c[row * N + col] = tmp;
    //printf("%d\n",tmp);
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
      //std::cout<<tmp<<"=="<<c[i*N+j]<<std::endl;
      // Check against the CPU result
      assert(tmp == c[i * N + j]);
    }
  }
}

int main(){
  
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

  int THREADS = 32;
  int BLOCKS = (int)ceil(N / THREADS);

  //std::cout<<"blocks: "<<BLOCKS*BLOCKS<<" threads: "<<THREADS*THREADS<<std::endl;

  dim3 threads(THREADS, THREADS);
  dim3 blocks(BLOCKS, BLOCKS);

  matrix_multiplication<<<blocks, threads>>>(d_a, d_b, d_c, N);

  cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

  verify_result(h_a, h_b, h_c);

  cout << "COMPLETED SUCCESSFULLY\n";

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}
