#include <iostream>
#include <algorithm>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
//#include <cutil.h>

#include<opencv2/opencv.hpp>
#include<fstream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void colorToGrey(unsigned char *rgbImage, unsigned char *greyImage, int width, int height){
    int col = (blockIdx.x * blockDim.x) + threadIdx.x;
    int row = (blockIdx.y * blockDim.y) + threadIdx.y;

    if(col<width && row<height){
        int greyOffset = row * width + col;

        int rgbOffset = 3 * greyOffset;

        unsigned char r = rgbImage[greyOffset];
        unsigned char g = rgbImage[greyOffset + 1];
        unsigned char b = rgbImage[greyOffset + 2];

        greyImage[greyOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Image file loader (RAW format)
////////////////////////////////////////////////////////////////////////////////
bool loadRawImage(char* filename, int w, int h, float* r, float* g, float* b)
{
	FILE *imageFile;
	imageFile = fopen(filename, "r");
    
	if (imageFile == NULL) 
	{
		printf("Cannot find texture file in data directory: %s\n", filename);
		return false;
	}
	else
	{	    
		for (int i = 0; i < h*w; i+=1)
		{
			r[i]	= fgetc(imageFile);
			g[i]	= fgetc(imageFile);
			b[i]	= fgetc(imageFile);
		}            
		
		fclose(imageFile);
		return true;
	}
}

////////////////////////////////////////////////////////////////////////////////
// Image file writer (RAW format)
////////////////////////////////////////////////////////////////////////////////
bool writeRawImage(char* filename, int w, int h, float* r, float* g, float* b)
{
	FILE *imageFile;
	imageFile = fopen(filename, "wb");
    
	if (imageFile == NULL) 
	{
		printf("Cannot write texture file: %s\n", filename);
		return false;
	}
	else
	{	  
		for (int i = 0; i < h*w; i+=1)
		{
			fputc((int)(r[i]), imageFile);
			fputc((int)(g[i]), imageFile);
			fputc((int)(b[i]), imageFile);
		}
		            
		fclose(imageFile);
		return true;
	}
    
}

int main(){
	float width, height;
	size_t data_size;
	float *h_DataR, *h_DataG, *h_DataB;
	float *h_ResultR, * h_ResultG, *h_ResultB;
	float *d_DataR, *d_DataG, *d_DataB, *d_Result;

	width = 1024;
	height = 1024;
	data_size = width * height * sizeof(float);

    char *iFilename = "../../Image/Input/girl1kby1k.jpg";
	char *oFilename = "./../Image/Output/girl1kby1k_out.jpg";

	// Memory Manage
	h_DataR		= (float *)malloc(data_size);
    h_DataG		= (float *)malloc(data_size);
    h_DataB		= (float *)malloc(data_size);

	cudaMalloc(&d_DataR, data_size);
	cudaMalloc(&d_DataG, data_size);
	cudaMalloc(&d_DataB, data_size);

	// Read raw image
	if (!loadRawImage(iFilename, width, height, h_DataR, h_DataG, h_DataB) )
    {
    	printf("File not found. random image generator will be used...\n");
	}

	// Image Process
	cudaMemcpy(d_DataR, h_DataR, data_size, cudaHostToDevice);
	cudaMemcpy(d_DataG, h_DataG, data_size, cudaHostToDevice);
	cudaMemcpy(d_DataB, h_DataB, data_size, cudaHostToDevice);

	//dim3 BLOCKS = 4*4;
	//dim3 THREADS = 256*256;
	//colorToGrey<<<BLOCKS, THREADS>>>();

	// Write result image
	h_ResultR	= (float *)malloc(data_size);
    h_ResultG	= (float *)malloc(data_size);
    h_ResultB	= (float *)malloc(data_size);

	writeRawImage(oFilename, width, height, h_ResultR, h_ResultG, h_ResultB);

}