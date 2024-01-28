// Haritha Selvakumaran 01/25/2024
// Header file for filter.cpp - contains function prototypes

#ifndef FILTER_H
#define FILTER_H

#include <opencv2/opencv.hpp>

// Tasks

int greyscale(cv::Mat &src, cv::Mat &dst);

int sepia(cv::Mat &src, cv::Mat &dst);

int blur5x5_1( cv::Mat &src, cv::Mat &dst );

int blur5x5_2( cv::Mat &src, cv::Mat &dst );

int blurQuantize( cv::Mat &src, cv::Mat &dst, int levels);

int sobelX3x3(cv::Mat &src, cv::Mat &dst);

int sobelY3x3(cv::Mat &src, cv::Mat &dst);

int magnitude( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst );

// Three more effects for the video

int emboss(cv::Mat &src, cv::Mat &dst); // Area effect

int inverse(cv::Mat &src, cv::Mat &dst); // Single-step pixel-wise modification effect

int colorFace(cv::Mat &src, cv::Mat &dst, std::vector<cv::Rect> &faces); // Effect built on face detection

// Extensions

int paintify(cv::Mat &src, cv::Mat &dst);

int texture(cv::Mat &src, cv::Mat &dst);

int cartoonize(cv::Mat &src, cv::Mat &dst);

#endif // FILTER_H
