#include "stdio.h"
#include<iostream>

#include<opencv2/core/core.hpp>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>

using namespace cv;

int main(){
    Mat img = imread("C:\\Users\\WangCheng\\Documents\\CUDA\\Image\\Input\\girl1kby1k.jpg\\girl1kby1k.jpg");
    namedWindow("test opencv");
    imshow("test opencv", img);
    waitKey(6000);
}