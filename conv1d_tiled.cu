#include"cuda_runtime.h"
#include"device_launch_parameters.h"

#include<iostream>
#include<cassert>
#include<cstdlib>
#include<vector>
#include<random>
#include<algorithm>
#include<functional>

#define TILE_SIZE 4

#define Mask_Width 5
__constant__ int d_M[Mask_Width];

__global__ void convolution_1D_tailed_kernel(int *N, int *P, int Width){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    //shared memory equal to tile size(4) + 2*(max_mask_width-1)/2, left halo size and right halo size. 
    __shared__ int N_ds[TILE_SIZE + Mask_Width - 1];

    int n = Mask_Width/2;

    //left halo index
    int halo_index_left = (blockIdx.x - 1)* blockDim.x + threadIdx.x;   //last block index

    if(threadIdx.x >= blockDim.x - n){  //idx >= dims - n then we need to add its left halo
        N_ds[threadIdx.x - (blockDim.x - n)] = (halo_index_left < 0)? 0 : N[halo_index_left];   //define left halo
    }

    N_ds[n + threadIdx.x] = N[blockIdx.x*blockDim.x + threadIdx.x]; //define tile size value

    //right halo index
    int halo_index_right = (blockIdx.x + 1)*blockDim.x + threadIdx.x;   //next block index

    if(threadIdx.x < n){  //idx < n then we need to add its right halo
        N_ds[n + blockDim.x + threadIdx.x] = (halo_index_right >= Width) ? 0 : N[halo_index_right]; //define right halo
    }

    //sync
    __syncthreads();

    int P_Value = 0;

    //thread calculate
    for(int j = 0; j < Mask_Width; j++){
        P_Value += N_ds[threadIdx.x + j] * d_M[j];
    }

    P[i] = P_Value;
}

void verify_result(int *array, int *mask, int *result, int n, int m) {
    int radius = m / 2;
    int temp;
    int start;
    for (int i = 0; i < n; i++) {
      start = i - radius;
      temp = 0;
      for (int j = 0; j < m; j++) {
        if ((start + j >= 0) && (start + j < n)) {
          temp += array[start + j] * mask[j];
        }
      }
      //std::cout<<temp<<"=="<<result[i]<<std::endl;
      assert(temp == result[i]);
    }
}

int main(){
    const int Width = 1<<10;    //input len
    //const int Mask_Width = 1<<5;    //filter len
    size_t W_bytes = sizeof(int)*Width;   //input mem size
    size_t M_W_bytes = sizeof(int)*Mask_Width;

    //Host
    std::vector<int> N(Width);
    std::vector<int> M(Mask_Width);
    std::vector<int> P(Width);
    std::generate(N.begin(), N.end(), [](){return rand()%100;});
    std::generate(M.begin(), M.end(), [](){return rand()%100;});

    //Device
    int *d_N, /*d_M,*/ *d_P;
    
    cudaMalloc(&d_N, W_bytes);
    //cudaMalloc(&d_M, M_W_bytes);
    cudaMalloc(&d_P, W_bytes);

    cudaMemcpy(d_N, N.data(), W_bytes, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_M, M.data(), M_W_bytes);

    //cudaDeviceSynchronize();

    int Threads = 32;
    int Blocks = (int)ceil(Width/Threads);

    convolution_1D_tailed_kernel<<<Blocks, Threads>>>(d_N, d_P, Width);

    //cudaDeviceSynchronize();

    cudaMemcpy(P.data(), d_P, W_bytes, cudaMemcpyDeviceToHost);

    //verify
    verify_result(N.data(), M.data(), P.data(), Width, Mask_Width);

    cudaFree(d_P);
    cudaFree(d_N);
    //cudaFree(d_M);
    
    
    std::cout<<"Completed Successfully!\n";

    return 0;
}