#include<iostream>
#include<cassert>
#include<cstdlib>
#include<vector>
#include<random>
#include<algorithm>
#include<functional>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"



__global__ void convolutional_1D_basic_kernel(int* N, int* M, int* P, int Width, int Mask_Width){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int Pvalue = 0;

    int N_start_point = i - (Mask_Width/2);

    for(int j=0; j<Mask_Width; j++){
        if((N_start_point + j) >= 0 && (N_start_point + j) < Width){
            Pvalue += N[N_start_point + j] * M[j];
        }
    }

    P[i] = Pvalue;
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
      assert(temp == result[i]);
    }
}

int main(){
    const int Width = 1<<10;    //input len
    const int Mask_Width = 1<<5;    //filter len
    size_t W_bytes = sizeof(int)*Width;   //input mem size
    size_t M_W_bytes = sizeof(int)*Mask_Width;

    //Host
    std::vector<int> N(Width);
    std::vector<int> M(Mask_Width);
    std::vector<int> P(Width);
    std::generate(N.begin(), N.end(), [](){return rand() % 100;});
    std::generate(M.begin(), M.end(), [](){return rand() % 100;});

    //Device
    int *d_N, *d_M, *d_P;
    
    cudaMalloc(&d_N, W_bytes);
    cudaMalloc(&d_M, M_W_bytes);
    cudaMalloc(&d_P, W_bytes);

    cudaMemcpy(d_N, N.data(), W_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_M, M.data(), M_W_bytes, cudaMemcpyHostToDevice);

    //cudaDeviceSynchronize();

    int Threads = 32;
    int Blocks = (int)ceil(Width/Threads);

    convolutional_1D_basic_kernel<<<Blocks, Threads>>>(d_N, d_M, d_P, Width, Mask_Width);

    //cudaDeviceSynchronize();

    cudaMemcpy(P.data(), d_P, W_bytes, cudaMemcpyDeviceToHost);

    //verify
    verify_result(N.data(), M.data(), P.data(), Width, Mask_Width);

    cudaFree(d_P);
    cudaFree(d_N);
    cudaFree(d_M);
    
    
    std::cout<<"Completed Successfully!\n";

    return 0;
}