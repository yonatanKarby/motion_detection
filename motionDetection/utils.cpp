#include "utils.h"


int getRectangleIndex(int index, int recW, int recH, int imageH, int imageW, int numberOfRectangels)
{
	int recHNums = imageH / recH;
	int recWNums = imageW / recW;

	int Windex = index % imageW;

	for (int rect = 0; rect < numberOfRectangels; rect++)
	{
		int heightIndex = ((rect / recWNums) + 1) * imageW * recH;
		if (index >= heightIndex) continue;

		int minW = (rect % recWNums) * recW;
		int maxW = minW + recW;
		
		if (Windex <= maxW && Windex >= minW)
		{
			return rect;
		}
	}
	return -1;
}

utils::utils(int rectangleWidth, int rectangleHeight, int imageWidth, int imageHeight)
{
	rectangleW = rectangleWidth;
	rectangleH = rectangleHeight;
	rectanglePixelCount = rectangleH * rectangleW;
	imageW = imageWidth;
	imageH = imageHeight;
}

utils::~utils()
{
	closeMotionDetectionThread(-1);
	delete imageQueue;
}

cv::Mat utils::imageDerivative(cv::Mat image)
{
	return image;
}

cv::Mat utils::getImage()
{
	cv::Mat image = imageQueue->pop();
	return image;
}

threadContext* utils::createThreadContext()
{
	threadContext* current = new threadContext();
	int numberOfRectangles = (imageH / rectangleH) * (imageW / rectangleW);
	for (int i = 0; i < numberOfRectangles; i++)
	{
		current->rectangles.push_back(0);
		current->recrangleMotionMap.push_back(0);
	}
	int length = imageW * imageH;
	for (int i = 0; i < length; i++)
	{
		int rectIndex = getRectangleIndex(i, rectangleW, rectangleH, imageH, imageW, numberOfRectangles);
		current->rectanglesMap.push_back(rectIndex);
	}
	return current;
}

void utils::addImage(cv::Mat image)
{
	imageQueue->push(image);
}

void threadLoop(utils* Utils, myThread* t)
{
	int counter = 0;
	cv::Mat currentImage;
	currentImage = Utils->getImage();

	Utils->setBaseImage(&currentImage, &t->baseImage);
	
	while (t->context->isRunning)
	{
		counter++;
		currentImage = Utils->getImage();

		if (currentImage.rows == 0 && currentImage.cols == 0)
			continue;
		cv::Mat motionImage = Utils->motionDetect(&currentImage, &t->baseImage ,*(t->context));

		if(Utils->displayWindow)
			cv::imshow("motion", motionImage);
		std::string path = "F:/yonatan/coding projects/2020/motionDetection/mainDir/ouput/ " + std::to_string((Utils->imageCounter)) +"t"+ std::to_string(t->threadId) + ".jpg";
		Utils->imageCounter++;
		cv::imwrite(path, motionImage);

		if (counter > 5)
		{
			Utils->setBaseImage(&currentImage, &t->baseImage);
			counter = 0;
		}
		cv::waitKey(1);
	}
}

void utils::openMotionDetectionThread(int threadCount, bool _displayWindow)
{
	displayWindow = _displayWindow;
	for (int i = 0; i < threadCount; i++)
	{
		myThread* t = new myThread();
		threads.push_back(t);
		int length = threads.size();
		t->threadId = length - 1;
		threads[length - 1]->context = createThreadContext();
		/////////////////thread begins//////////////////////
		threads[length - 1]->thr = new std::thread(threadLoop, this, t);
	}
}

void utils::closeMotionDetectionThread(int threadIndex)
{
	if (threadIndex == -1)
	{
		int length = threads.size();
		for (int i=0; i<length; i++)
		{
			threads[i]->context->isRunning = false;
			threads[i]->thr->join();

			delete threads[i]->thr;
			delete threads[i]->context;
		}
		return;
	}
	threads[threadIndex]->context->isRunning = false;
	threads[threadIndex]->thr->join();

	delete threads[threadIndex]->thr;
	delete threads[threadIndex]->context;
}

void utils::openMotionDetector(int threadCount, bool _displayWindow)
{
	openMotionDetectionThread(threadCount, _displayWindow);
}

cv::Mat utils::toGrayScale(cv::Mat* image)
{
	cv::Mat returnImage;
	cv::cvtColor((*image), returnImage, cv::COLOR_BGR2GRAY);
	return returnImage;
}
void utils::setBaseImage(cv::Mat* image, cv::Mat* basePtr)
{
	cv::Mat gray = toGrayScale(image);
	*basePtr = edgeDetection(&gray);
}

cv::Mat utils::motionDetect(cv::Mat* newImage,cv::Mat* baseImage, threadContext context)
{
	cv::Mat gray = toGrayScale(newImage);
	cv::Mat edged = edgeDetection(&gray);
	cv::Mat motion = subtractMotion(&edged, baseImage, threshHold, context);
	
	int length = context.rectangles.size();
	int min = 255 * rectanglePixelCount;
	for (int i = 0; i < length; i++)
	{
		if (context.rectangles[i] < min)
			min = context.rectangles[i];
	}
	threshHold = min;
	for (int i = 0; i < length; i++)
	{
		if (context.rectangles[i] > threshHold)
		{
			context.recrangleMotionMap[i] = 1;
		}
		else
		{
			context.recrangleMotionMap[i] = 0;
		}
		context.rectangles[i] = 0;
	}
	return motion;
}

cv::Mat utils::subtractMotion(cv::Mat* image1, cv::Mat* image2, int pixelThreshHold, threadContext context)
{
	int width = image1->cols;
	int height = image2->rows;

	cv::Mat newImage;
	image1->copyTo(newImage);

	auto image1ptr = image1->ptr();
	auto image2ptr = image2->ptr();
	auto newImagePtr = newImage.ptr();

	int length = width * height;

	for(int i=0; i<length; i++)
	{
		char newPixel = image1ptr[0] - image2ptr[0];
		
		if (newPixel < 0) { newPixel = 0; }
		newImagePtr[0] = newPixel;

		context.rectangles[context.rectanglesMap[i]] += newPixel;

		newImagePtr++;
		image1ptr++;
		image2ptr++;
	}
	return newImage;
}

cv::Mat utils::edgeDetection(cv::Mat* image, threadContext context)
{
	//pre process

	int matrixHeight = 3;
	int matrixWidth = 3;

	int midMatrixWidth = (matrixWidth / 2.0) + 1;
	int midMatrixHeight = (matrixHeight / 2.0) + 1;

	double midWidth = (matrixWidth / 2.0);
	double midHeight = (matrixHeight / 2.0);

	int width = image->cols;
	int height = image->rows;

	cv::Mat newImage;
	image->copyTo(newImage);

	int maxX = width - midMatrixWidth;
	int maxY = height - midMatrixHeight;

	int matrixSize = context.indexOffsets.size();

	//matrix calculations

	int length = width * height - (midMatrixHeight * width); //needs fix
	
	auto imagePtr = image->ptr<uchar>();
	auto newImagePtr = newImage.ptr<uchar>();

	imagePtr += midMatrixHeight * width;
	newImagePtr += midMatrixHeight * width;

	//loop through all the image pixels
	for(int ptrIndex = 0; ptrIndex < length; ptrIndex++)
	{
		int matrixSum = 0;
		for (int i = 0; i < matrixSize; i++)
		{
			int offset = context.indexOffsets[i];
			
			if (context.matrixVals[i] > 0)
			{
				matrixSum += imagePtr[offset] >> 1;
			}
			else
			{
				matrixSum -= imagePtr[offset] >> 1;
			}
		}
		if (matrixSum < threshHold || matrixSum > (255 - threshHold))
			matrixSum = 0;

		newImagePtr[0] = matrixSum;
		
		imagePtr++;
		newImagePtr++;
	}
	return newImage;
}
