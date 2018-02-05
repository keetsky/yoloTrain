/****
将视频的时间序列保存为图片序列
  video              imgjpg     
saobin.mp4  ->     2000000.jpg
                   2000001.jpg
                      ... 
*/

#include <cv.h>  
#include <highgui.h>  
#include <opencv2/opencv.hpp>
#include<iostream>
#include<fstream>
#include<string>

#include<stdio.h>
using namespace cv;  
 
int main(int argc, char* argv[])  
{  
	
	VideoCapture capture("saobin.mp4");
	if(!capture.isOpened())
	{
	    printf("open error!\n");
	    return -1;	
	}
	//capture.open(0);
	//capture.set(CV_CAP_PROP_FRAME_WIDTH, 1920);  
   	 //capture.set(CV_CAP_PROP_FRAME_HEIGHT, 1080); 
	
        int fnumb=200000;
        string filename("imgjpg/");
	namedWindow("视频",CV_NORMAL); 
	while(1)
	{
		Mat frame;
		capture>>frame;
		imshow("视频",frame);
                string imgname=".jpg";
                
                imgname=filename+std::to_string(fnumb)+imgname;
                imwrite(imgname,frame);
                fnumb++;
		waitKey(30);
		//std::cout<<frame.cols<<" "<<frame.rows<<std::endl;
	}
   // namedWindow("Display Image", CV_WINDOW_AUTOSIZE);  
  
   
    return 0;  
}




