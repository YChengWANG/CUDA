#include<stdio.h>
#include<cassert>
#include<iostream>

__global__ void vectorAdd(int* a, int* b, int* c, int N){
    int tid = (BlockDims.x * BlockIdx.x) + ThreadIdx.x

    if(tid < N) c[tid] = a[tid] + b[tid];
}

int main(){
    const int N = 1<<16;
    size_t bytes = sizeof(int) * N;

    int *a, *b, *c;

    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    int id = cudaGetDevice(&id);

    cudaMemAdvise(a, bytes, cudaMemAdviseSetPreferredLoction, cudaCpuDeviceId);
    cudaMemAdvise(b, bytes, cudaMemAdviseSetPreferredLoction, cudaCpuDeviceId);
    cudaMemPrefetchAsyc(c, bytes, id);

    for(int i = 0; i < N; i++){
        a[i] = rand() % 100;
        b[i] = rand() % 100;
    }

    cudaMemAdvise(a, bytes, cudaMemAdviseSetReadMostly, id);
    cudaMemAdvise(b, bytes, cudaMemAdviseSetReadMostly, id);
    cudaMemPrefetchAsync(a, bytes, id);
    cudaMemPrefetchAsync(b, bytes, id);

    int BLOCK_SIZE = 1<<10;

    int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    vectorAdd<<<GRID_SIZE, BLOCK_SIZE>>>(a, b, c, N);

    cudaDeviceSynchronize();

    cudaMemPrefetchAsync(a, bytes, cudaCpuDeviceId);
    cudaMemPrefetchAsync(b, bytes, cudaCpuDeviceId);
    cudaMemPrefetchAsync(c, bytes, cudaCpuDeviceId);

    for(int i = 0; i < N; i++){
        assert(c[i] = a[i] + b[i]);
    }

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    std::cout<<"Completed Successfully!\n";

    return 0;
}