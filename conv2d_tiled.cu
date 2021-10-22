#include<iostream>
#include<cassert>
#include<cstdlib>
#include<vector>
#include<random>
#include<algorithm>
#include<functional>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define MASK_WIDTH 7
__constant__ int d_M[MASK_WIDTH*MASK_WIDTH];

#define TILE_SIZE 1<<9; // Width/2

__global__ void convolutional_2D_tiled_kernel(int *d_N, int *d_P, int Width){
    
    // declare row and col
    int row = (blockIdx.y * blockDim.y) + threadIdx.y;
    int col = (blockIdx.x * blockDim.x) + threadIdx.x;

    //share memory
    __shared__ int SHMEM[(TILE_SIZE + MASK_WIDTH - 1)*(TILE_SIZE + MASK_WIDTH - 1)];

    // declare halo
    // upper left
    


    // sync
    __syncthreads()

}

void verify_result(int *m, int *mask, int *result, int N, int M) {
    // Temp value for accumulating results
    int temp;
  
    // Intermediate value for more readable code
    int offset_r;
    int offset_c;
  
    // Go over each row
    for (int i = 0; i < N; i++) {
      // Go over each column
        for (int j = 0; j < N; j++) {
            // Reset the temp variable
            temp = 0;

            // Go over each mask row
            for (int k = 0; k < M; k++) {
                // Update offset value for row
                offset_r = i - M/2 + k;

                // Go over each mask column
                for (int l = 0; l < M; l++) {
                    // Update offset value for column
                    offset_c = j - M/2 + l;

                    // Range checks if we are hanging off the matrix
                    if (offset_r >= 0 && offset_r < N) {
                        if (offset_c >= 0 && offset_c < N) {
                            // Accumulate partial results
                            temp += m[offset_r * N + offset_c] * mask[k * M + l];
                        }
                    }
                }
            }
            
            std::cout<<result[i * N + j]<<"=="<<temp<<std::endl;
            // Fail if the results don't match
            assert(result[i * N + j] == temp);
        }
    }
}