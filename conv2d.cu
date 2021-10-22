#include<iostream>
#include<cassert>
#include<cstdlib>
#include<vector>
#include<random>
#include<algorithm>
#include<functional>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


__global__ void convolutional_2D_basic_kernel(int *N, int *M, int *P, int Width, int Mask_Width){

    //init
    int Pval = 0;
    
    //declare which thread we use
    int tidy = blockIdx.y*blockDim.y + threadIdx.y;
    int tidx = blockIdx.x*blockDim.x + threadIdx.x;
    
    //print
    //printf("%d || %d",tidy,tidx);
    
    //start point
    int N_start_point_y = tidy - (Mask_Width/2);
    int N_start_point_x = tidx - (Mask_Width/2);

    //calculate pvalue
    for(int row=0;row<Mask_Width;row++){
        for(int col=0;col<Mask_Width;col++){
            if(
                ((N_start_point_x + col) >= 0 && (N_start_point_x + col)<Width) &&
                ((N_start_point_y + row) >= 0 && (N_start_point_y + row)<Width)
            ){
                //printf("is in\n");
                //printf("%d, %d\n",N[(N_start_point_y + row) * Width + (N_start_point_x + col)], M[row * Mask_Width + col]);
                Pval += N[(N_start_point_y + row) * Width + (N_start_point_x + col)] * M[row * Mask_Width + col];
            }
        }
    }

    P[tidy * Width + tidx] = Pval;
    //printf("%d, %d == %d\n",tidy,tidx,Pval);
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

int main(){
    const int Width = 1<<10;    //input len
    const int Mask_Width = 5;    //filter len

    size_t W_bytes = sizeof(int)*Width*Width;   //input mem size
    size_t M_W_bytes = sizeof(int)*Mask_Width*Mask_Width;   //input mask mem size

    //Host
    std::vector<int> N(Width*Width);
    std::vector<int> M(Mask_Width*Mask_Width);
    std::vector<int> P(Width*Width);

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

    int THREADS = 16;
    int BLOCKS = (int)ceil(Width/THREADS);

    dim3 Threads(THREADS,THREADS);
    dim3 Blocks(BLOCKS,BLOCKS);

    convolutional_2D_basic_kernel<<<Blocks, Threads>>>(d_N, d_M, d_P, Width, Mask_Width);

    cudaDeviceSynchronize();

    cudaMemcpy(P.data(), d_P, W_bytes, cudaMemcpyDeviceToHost);
    
    //print
    //for(int p=0; p<P.size()-1; p++){
    //    std::cout<<P[p]<<std::endl;
    //}
    
    verify_result(N.data(), M.data(), P.data(), Width, Mask_Width);

    cudaFree(d_N);
    cudaFree(d_M);
    cudaFree(d_P);
    
    std::cout<<"Completed Successfully!\n";

    return 0;
}