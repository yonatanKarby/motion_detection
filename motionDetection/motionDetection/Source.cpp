#include <iostream>
#include "opencv2/opencv.hpp";
#include "utils.h";

std::shared_ptr<utils> Utils = std::make_shared<utils>(64,48,640,480);
cv::VideoCapture cap;


int main()
{
	if (!cap.open(0))
		return -1;

	cv::Mat image;
	cap >> image;
	
	Utils->openMotionDetector(&cap);

	while (true)
	{
		std::cout << Utils->isMotion << std::endl;
		cv::waitKey(1);
	}
	std::cin.get();
}