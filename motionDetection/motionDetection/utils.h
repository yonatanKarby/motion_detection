#pragma once
#include <iostream>
#include <stdint.h>
#include <chrono>
#include <string>
#include "opencv2/opencv.hpp";

struct matrix
{
	std::vector<std::vector<float>> _matrix;
	matrix(std::vector<std::vector<float>> newMatrix)
	{
		_matrix = newMatrix;
	}
};

struct threadContext
{
	std::vector<int> rectangles = std::vector<int>();
	std::vector<int> recrangleMotionMap = std::vector<int>();
	std::vector<int> rectanglesMap = std::vector<int>();


	bool isRunning = true;

	threadContext()
	{
	};
};

struct myThread
{
	threadContext* context;
	void* Utils;
	cv::Mat baseImage;

	myThread() {};
	myThread(const myThread &c) 
	{
		context = c.context;
	};
	~myThread()
	{
		delete context;
	}
};

#define NOW_TIME std::chrono::high_resolution_clock::now()

static auto timeout = std::chrono::seconds(100000000000000);


template<typename T>
class Queue
{
private:
	std::vector<T> queue = std::vector<T>();
	
public:
	Queue() {};

	void push(T newObject)
	{
		queue.push_back(newObject);
	}

	T pop()
	{
		while (queue.size() <=0 )
		{
		}
		T poped = queue[0];
		queue.erase(queue.begin());
		return poped;
	}

	bool empty()
	{
		if (queue.size() <= 0)
			return true;
		return false;
	}

	void print()
	{
		int length = queue.size();
		for (int i = 0; i < length; i++)
		{
			std::cout << queue[i] << "->";
		}
		std::cout << std::endl;
	}
};

class utils
{
public:
	utils(int rectangleWidth, int rectangleHeight, int imageWidth, int imageHeight);
	~utils();

	matrix gx = matrix({
		{1, 0, 0},
		{0, 0, 0},
		{0, 0, -1}
	});
	
	bool displayWindow = false;

	int imageCounter = 0;
	int rectangleW;
	int rectangleH;
	int imageW;
	int imageH;
	int rectanglePixelCount;
	int rectangleCount;

	std::thread* mainThread;
	cv::VideoCapture* myCap;
	myThread* t;

	std::vector<int> matrixVals = { 1, -1 };
	std::vector<int> indexOffsets = { -641, 641 };

	bool isMotion = false;
	double threshHold = 0;
	int edgeThreshHold = 10;
	int motionThreshhold = 21;
	double currentImageMotionVal = 0;

	cv::Mat imageDerivative(cv::Mat image);
	cv::Mat edgeDetection(cv::Mat* image);
	cv::Mat toGrayScale(std::shared_ptr<cv::Mat> image);

	std::shared_ptr<cv::Mat> getImage();

	threadContext* createThreadContext();

	void processImage(threadContext context, cv::Mat image);

	void setBaseImage(std::shared_ptr<cv::Mat> image, cv::Mat* basePtr);
	cv::Mat subtractMotion(cv::Mat* image1, cv::Mat* image2, int threshHold, threadContext context);
	cv::Mat motionDetect(std::shared_ptr<cv::Mat> newImage, cv::Mat* baseImage, threadContext context);

	void openMotionDetector(cv::VideoCapture* cap);
};