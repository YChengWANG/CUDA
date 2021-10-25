#include "stdio.h"
#include<iostream>

#include<opencv2/core/core.hpp>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>

using namespace cv;

int main(){
    Mat img = imread("./Input/girl1kby1k.jpg");
    namedWindow("test opencv");
    imshow("test opencv", img);
    waitKey(6000);
}