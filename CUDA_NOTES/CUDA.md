# **Programming Massively Parallel Processor NOTES**

## **GPU structure**

- **Terminology：**
  - Streaming Multiprocessors (SMs) - BLOCK
  - Streaming Processors (SPs) - THREAD
  - Synchronous DRAM (SDRAM) - Global MEM
  - Graphics Double Data Rate (GDDR) - GDDR SRAMs(used for graphics)
  - High-Bandwidth Memory (HBM)
  - Single Instruction Multiple Data (SIMD)

![](../CUDA_NOTES/IMG/Snipaste_2021-10-21_15-47-47.png)

## CUDA compliation

- NVIDIA C Complier(NVCC)
  
![](../CUDA_NOTES/IMG/Snipaste_2021-10-21_16-42-05.png)
![](../CUDA_NOTES/IMG/Snipaste_2021-10-21_16-56-03.png)
![](../CUDA_NOTES/IMG/Snipaste_2021-10-21_17-19-37.png)

## **Memory and data locality**

- Registers (thread R/W)
- Local Memory (thread R/W)
- Shared Memory / Scratchpad memory (block R/W)
- Global Memory (grid R/W)
- Constant Memory (grid R only)
  
![](../CUDA_NOTES/IMG/Snipaste_2021-10-22_09-55-00.png)

- arithmetic and logic unit (ALU)
![](../CUDA_NOTES/IMG/Snipaste_2021-10-22_10-00-37.png)
