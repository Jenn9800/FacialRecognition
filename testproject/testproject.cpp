// testproject.cpp : 此檔案包含 'main' 函式。程式會於該處開始執行及結束執行。
//
#define NOMINMAX//add

#include <opencv2/core/core.hpp>//opencv
#include <opencv2/highgui/highgui.hpp>//opencv 
#include <opencv2/imgproc/imgproc.hpp>//opencv 
#include <opencv2/opencv.hpp>//opencv 
#include <opencv2/dnn.hpp>
#include <stdio.h>
#include <iostream>//c++
#include <Windows.h>
#include <ctype.h>
#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <math.h>
#include <fstream>  
#include <time.h>
#include <string>


using namespace std;//c++
using namespace cv;//opencv 

CascadeClassifier cascade;
Mat frame, frameCopy, grayFrame;
Mat AdaBoostFaceDetect(Mat inputFrame);
Mat dnnFaceDetect(Mat inputFrame);
dnn::Net net;

#include <string>
#include <filesystem> // Include the filesystem library

#include <iostream>
#include <string>

#ifdef _WIN32
#include <direct.h>
#define CREATE_DIR(dir) _mkdir(dir)
#else
#include <sys/stat.h>
#define CREATE_DIR(dir) mkdir(dir, 0777)
#endif

using namespace std;

int main() {
	cascade.load("haarcascade_frontalface_alt.xml");
	net = dnn::readNetFromTensorflow("opencv_face_detector_uint8.pb", "opencv_face_detector.pbtxt");

	float adaRate = 2.0 / 7.0;
	cout << "Ada Boost: " << adaRate * 100 << "%" << endl;
	float dnnRate = 5.0 / 7.0;
	cout << "Dnn Boost: " << dnnRate * 100 << "%" << endl;

	// Create directories for output images if they don't exist
	CREATE_DIR("AdaBoost");
	CREATE_DIR("DNN");

	for (int i = 1; i <= 10; ++i) {
		// Construct the input image file name
		string inputFileName = to_string(i) + ".jpg";

		// Read the input image
		frame = imread(inputFileName);
		if (frame.empty()) {
			cerr << "Error reading image: " << inputFileName << endl;
			continue;  // Skip to the next iteration if the image cannot be read
		}

		frame.copyTo(frameCopy);
		cvtColor(frameCopy, grayFrame, CV_BGR2GRAY);

		// Perform face detection using AdaBoost
		AdaBoostFaceDetect(grayFrame);

		// Save the output image to the AdaBoost folder
		imwrite("AdaBoost/output_" + to_string(i) + ".jpg", frameCopy);

		// Perform face detection using DNN
		dnnFaceDetect(frame);

		// Save the output image to the DNN folder
		imwrite("DNN/output_" + to_string(i) + ".jpg", frame);

		waitKey(0); // Wait for a key press before moving to the next image
	}

	system("pause");
	return 0;
}


Mat AdaBoostFaceDetect(Mat inputFrame)//input gray image(mat format), return color image copy(mat format)
{
	vector<Rect> rect;

	//color for mark face
	Scalar colors[] =
	{
		CV_RGB(0,0,255),
		CV_RGB(0,128,255),
		CV_RGB(0,255,255),
		CV_RGB(0,255,0),
		CV_RGB(255,128,0),
		CV_RGB(255,255,0),
		CV_RGB(255,0,0),
		CV_RGB(255,0,255)
	};

	cascade.detectMultiScale(grayFrame, rect, 1.1, 2, CV_HAAR_DO_CANNY_PRUNING, cvSize(30, 30));//face detect

	for (int i = 0; i < rect.size(); i++)
	{
		Rect box(rect[i].x, rect[i].y, rect[i].width, rect[i].height);//x coordinate, y coordinate, width, height
		rectangle(frameCopy, box, Scalar(0, 0, 255), 2, 8, 0);//draw rectangle of face image
	}
	return frameCopy;
}

Mat dnnFaceDetect(Mat inputFrame)//input image(Mat format), return image(Mat format)
{
	Mat blob, probs;//Mat format, automatic release memory

	blob = dnn::blobFromImage(inputFrame, 1.0, Size(300, 300), Scalar(104.0, 177.0, 123.0), false, false);//image preprocessing
	//(input image, scale value, learning size, RGB minus value, if switch RB, if crop image)
	net.setInput(blob);//result put back to net
	probs = net.forward();//NCHW, 4-dimension matrix, 1: input image ID, 2: input image number, 3: each detect object, 4: each detect object info.
	Mat detectionMat(probs.size[2], probs.size[3], CV_32F, probs.ptr<float>());//only need last 2 dimensions info

	for (int i = 0; i < detectionMat.rows; i++)//info set 
	{
		float confidence = detectionMat.at<float>(i, 2);
		if (confidence > 0.5)//confidence: image score, close to 1 more similar to traning modle 
		{
			size_t objIndex = (size_t)(detectionMat.at<float>(i, 1));
			int x1 = detectionMat.at<float>(i, 3) * inputFrame.cols;//create rectangle for face region
			int y1 = detectionMat.at<float>(i, 4) * inputFrame.rows;//create rectangle for face region
			int x2 = detectionMat.at<float>(i, 5) * inputFrame.cols;//create rectangle for face region
			int y2 = detectionMat.at<float>(i, 6) * inputFrame.rows;//create rectangle for face region

			Rect box((int)x1, (int)y1, (int)(x2 - x1), (int)(y2 - y1));//x coordinate, y coordinate, width, height
			rectangle(inputFrame, box, Scalar(0, 0, 255), 2, 8, 0);
		}
	}
	return inputFrame;
}





