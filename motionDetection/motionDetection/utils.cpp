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
	delete mainThread;
}

cv::Mat utils::imageDerivative(cv::Mat image)
{
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

std::shared_ptr<cv::Mat> utils::getImage()
{
	std::shared_ptr<cv::Mat> image = std::make_shared<cv::Mat>();
	(*myCap) >> (*image);
	return image;
}

double calculateAvg(std::vector<double> arr)
{
	int length = arr.size();
	int sum = 0;
	for (auto element : arr)
		sum += element;
	return sum / length;
}

void* threadLoop(void* arg)
{
	myThread* t = (myThread*)arg;
	utils* Utils = (utils*)t->Utils;
	int counter = 0;
	std::shared_ptr<cv::Mat> currentImage;
	currentImage = Utils->getImage();

	Utils->setBaseImage(currentImage, &t->baseImage);

	int motionAvgLength = 16;
	double overMotion = 1.0 / motionAvgLength;
	std::vector<double> motionAvgs = std::vector<double>();
	for (int i = 0; i < motionAvgLength; i++)
		motionAvgs.push_back(0);
	int motionAvgIndex = 0;

	while (t->context->isRunning)
	{
		counter++;

		currentImage = Utils->getImage();

		if (currentImage->rows == 0 && currentImage->cols == 0)
			continue;

		cv::Mat motionImage = Utils->motionDetect(currentImage, &t->baseImage ,*(t->context));
		

		
		if (Utils->currentImageMotionVal > Utils->threshHold)
			Utils->isMotion = true;
		else
			Utils->isMotion = false;
		
		motionAvgs[motionAvgIndex] = Utils->currentImageMotionVal;//add the new value to the avgs list
		motionAvgIndex = (motionAvgIndex + 1) & 0xf;
		
		Utils->threshHold = 0;
		
		for (int i = 0; i < motionAvgLength; i++)
			Utils->threshHold += motionAvgs[i];
		
		Utils->threshHold *= overMotion;



		//show the image
		cv::imshow("motion", motionImage);
		if (counter > 5)
		{


			Utils->setBaseImage(currentImage, &t->baseImage);
			counter = 0;
		}
		cv::waitKey(1);
	}
	return nullptr;
}

void utils::openMotionDetector(cv::VideoCapture* cap)
{
	myCap = cap;
	t = new myThread();
	t->Utils = this;
	t->context = createThreadContext();
	mainThread = new std::thread(threadLoop, (void*)t);
}

cv::Mat utils::toGrayScale(std::shared_ptr<cv::Mat> image)
{
	cv::Mat returnImage;
	cv::cvtColor((*image), returnImage, cv::COLOR_BGR2GRAY);
	return returnImage;
}
void utils::setBaseImage(std::shared_ptr<cv::Mat> image, cv::Mat* basePtr)
{
	cv::Mat gray = toGrayScale(image);
	*basePtr = edgeDetection(&gray);
}

cv::Mat utils::motionDetect(std::shared_ptr<cv::Mat> newImage,cv::Mat* baseImage, threadContext context)
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
	//over all 2N passes on the image
	int width = image1->cols;
	int height = image2->rows;

	
	cv::Mat newImage;
	image1->copyTo(newImage);// this is N on the image

	auto image1ptr = image1->ptr();
	auto image2ptr = image2->ptr();
	auto newImagePtr = newImage.ptr();

	int length = width * height;

	currentImageMotionVal = 0.0;

	for(int i=0; i<length; i++)//this is another N pass on the image
	{
		char newPixel = image1ptr[0] - image2ptr[0];
		
		if (newPixel < 0) 
		{
			newPixel = 0; 
		}
		if (newPixel > 40)
		{
			double dpixel = (double)newPixel;
			currentImageMotionVal += (dpixel * dpixel);
		}
		newImagePtr[0] = newPixel;
		
		context.rectangles[context.rectanglesMap[i]] += newPixel;

		newImagePtr++;
		image1ptr++;
		image2ptr++;
	}
	return newImage;
}

cv::Mat utils::edgeDetection(cv::Mat* image)
{
	//pre process
	int matrixHeight = gx._matrix.size();
	int matrixWidth = gx._matrix[0].size();

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
	int matrixSize = indexOffsets.size();

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
			int offset = indexOffsets[i];
			
			if (matrixVals[i] > 0)
			{
				matrixSum += imagePtr[offset] >> 1;
			}
			else
			{
				matrixSum -= imagePtr[offset] >> 1;
			}
		}
		if (matrixSum < edgeThreshHold || matrixSum > (255 - edgeThreshHold))
			matrixSum = 0;

		newImagePtr[0] = matrixSum;
		
		imagePtr++;
		newImagePtr++;
	}
	return newImage;
}
