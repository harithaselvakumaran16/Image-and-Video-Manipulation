// Haritha Selvakumaran 01/23/2020
// Purpose: Contains all the image manipulation functions for the class

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include "filter.h"

using namespace cv;
using namespace std;

// Purpose: Converts a color image to greyscale using custom algorithm
/*
 * @param src: source image
 * @param dst: destination image
 * @return 0 if successful
 */

int greyscale(Mat &src, Mat &dst)
{
    dst = src.clone();
    for (int row = 0; row < src.rows; row++) {
        for (int col = 0; col < src.cols; col++) {
            Vec3b pixel = src.at<cv::Vec3b>(row, col);
            // Create grayscale by inverting colors and assigning the average of the inverted colors to each channel
            // uchar gray = static_cast<uchar>((255 - pixel[0] + 255 - pixel[1] + 255 - pixel[2])/3);
            uchar gray = static_cast<uchar>(0.5 * pixel[0] + 0.3 * pixel[1] + 0.4 * pixel[2]);
            dst.at<Vec3b>(row, col) = Vec3b(gray, gray, gray);
        }
    }

    return 0;
}

/*  Purpose: Converts a color image to sepia using custom filter
    * @param src: source image
    * @param dst: destination image
    * @return 0 if successful
*/

int sepia(Mat &src, Mat &dst)
{
    dst = src.clone();
    for (int row=0; row < src.rows; row++) {
        for (int col = 0; col < src.cols; col++) {
            Vec3b pixel = src.at<Vec3b>(row, col);

            uchar blue = pixel[0];
            uchar green = pixel[1];
            uchar red = pixel[2];

            float new_red = 0.272 * red + 0.534 * green + 0.131 * blue;
            float new_green = 0.349 * red + 0.686 * green + 0.168 * blue;
            float new_blue = 0.393 * red + 0.769 * green + 0.189 * blue;

            new_red = min(round(new_red), 255.0f);
            new_green = min(round(new_green), 255.0f);
            new_blue = min(round(new_blue), 255.0f);

            dst.at<Vec3b>(row, col) = Vec3b((uchar)new_red, (uchar)new_green, (uchar)new_blue);
            // putText(dst, "Sepia", Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2);
        }
    }
    return 0;
}

/*  Purpose: Blurs an image using a 5x5 Gaussian kernel
    * @param src: source image
    * @param dst: destination image
    * @return 0 if successful
*/

int blur5x5_1(Mat &src, Mat &dst ) {
    dst = src.clone();
    int kernel[5][5] = {
      {1, 2, 4, 2, 1},
      {2, 4, 8, 4, 2},
      {4, 8, 16, 8, 4},
      {2, 4, 8, 4, 2},
      {1, 2, 4, 2, 1}
    };
    int blue = 0, green = 0, red = 0;
    for (int y = 2; y < src.rows - 2; y++) {
        for (int x = 2; x < src.cols - 2; x++) {
            // Initialize values for each color channel
            float blue = 0, green = 0, red = 0;

            // Apply the blur filter using the Gaussian kernel
            for (int ky = -2; ky <= 2; ky++) {
                for (int kx = -2; kx <= 2; kx++) {
                    blue += src.at<Vec3b>(y + ky, x + kx)[0] * kernel[ky + 2][kx + 2];
                    green += src.at<Vec3b>(y + ky, x + kx)[1] * kernel[ky + 2][kx + 2];
                    red += src.at<Vec3b>(y + ky, x + kx)[2] * kernel[ky + 2][kx + 2];
                }
            }

            // Normalize the values
            blue /= 80.0f;
            green /= 80.0f;
            red /= 80.0f;

            // Update the corresponding pixel in the destination image
            dst.at<Vec3b>(y, x) = Vec3b(saturate_cast<uchar>(blue),
                                                saturate_cast<uchar>(green),
                                                saturate_cast<uchar>(red));
        }
    }

    return 0; // Success
  }

/*  Purpose: Blurs an image using a 5x5 Gaussian kernel but faster as it uses seperable filters 
    * @param src: source image
    * @param dst: destination image
    * @return 0 if successful
*/

int blur5x5_2(Mat &src, Mat &dst) {
    dst = src.clone();
    int kernel[5] = {1, 4, 6, 4, 1};
    int kernelSum = 16; // Sum of all the elements in the kernel

    // Horizontal pass
    for(int y = 0; y < src.rows; y++) {
        for(int x = 2; x < src.cols - 2; x++) {
            Vec3f sum = Vec3f(0, 0, 0);
            for(int kx = -2; kx <= 2; kx++) {
                Vec3b pixel = src.ptr<Vec3b>(y)[x + kx];
                for(int i = 0; i < 3; i++) {
                    sum[i] += pixel[i] * kernel[kx + 2];
                }
            }
            Vec3b avg;
            for(int i = 0; i < 3; i++) {
                avg[i] = saturate_cast<uchar>(sum[i] / kernelSum);
            }
            dst.ptr<Vec3b>(y)[x] = avg;
        }
    }

    // Vertical pass
    Mat temp = dst.clone();
    for(int y = 2; y < src.rows - 2; y++) {
        for(int x = 0; x < src.cols; x++) {
            Vec3f sum = Vec3f(0, 0, 0);
            for(int ky = -2; ky <= 2; ky++) {
                Vec3b pixel = temp.ptr<Vec3b>(y + ky)[x];
                for(int i = 0; i < 3; i++) {
                    sum[i] += pixel[i] * kernel[ky + 2];
                }
            }
            Vec3b avg;
            for(int i = 0; i < 3; i++) {
                avg[i] = saturate_cast<uchar>(sum[i] / kernelSum);
            }
            dst.ptr<Vec3b>(y)[x] = avg;
        }
    }
    return 0;
  }

/*  Purpose: Applies a 3x3 Sobel filter in the X direction
    * @param src: source image
    * @param dst: destination image
    * @return 0 if successful
*/

int sobelX3x3(Mat &src, Mat &dst) {
    dst.create(src.size(), CV_16SC3);

    Mat kernel = (Mat_<float>(3,3) << -1, 0, 1,
                                      -2, 0, 2,
                                      -1, 0, 1);
    for (int row=1; row < src.rows - 1; row++) {
        for (int col = 1; col < src.cols - 1; col++) {
            Vec3s sum = Vec3s(0, 0, 0);
            for (int i = -1; i <= 1; i++) {
                for (int j=-1; j <= 1; j++) {
                    // if (row + i >= 0 && row + i < src.rows && col + j >= 0 && col + j < src.cols) {
                        Vec3b pixel = src.at<Vec3b>(row + i, col + j);
                        for (int k = 0; k < 3; k++) {
                            sum[k] += pixel[k] * kernel.at<float>(i + 1, j + 1);
                        }
                    }
                }
            dst.at<Vec3s>(row, col) = sum;
        }
    }

    return 0;
}

/*  Purpose: Applies a 3x3 Sobel filter in the Y direction
    * @param src: source image
    * @param dst: destination image
    * @return 0 if successful
*/

int sobelY3x3(Mat &src, Mat &dst) {
    dst.create(src.size(), CV_16SC3);

    Mat kernel = (Mat_<float>(3,3) << -1, -2, -1,
                                      0, 0, 0,
                                      1, 2, 1);
    for (int row=1; row < src.rows - 1; row++) {
        for (int col = 1; col < src.cols - 1; col++) {
            Vec3s sum = Vec3s(0, 0, 0);
            for (int i = -1; i <= 1; i++) {
                for (int j=-1; j <= 1; j++) {
                    // if (row + i >= 0 && row + i < src.rows && col + j >= 0 && col + j < src.cols) {
                        Vec3b pixel = src.at<Vec3b>(row + i, col + j);
                        for (int k = 0; k < 3; k++) {
                            sum[k] += pixel[k] * kernel.at<float>(i + 1, j + 1);
                        }
                    }
            }
            dst.at<Vec3s>(row, col) = sum;
        }
    }

    return 0;
}

/*  Purpose: Calculates the magnitude of the gradient of an image
    * @param sx: sobel image in the X direction
    * @param sy: sobel image in the Y direction
    * @param dst: destination image
    * @return 0 if successful
*/

int magnitude(Mat &sx, Mat &sy, Mat &dst) {
    dst.create(sx.size(), CV_8UC3);

    for(int row = 0; row < sx.rows; row++) {
        for(int col = 0; col < sx.cols; col++) {
            Vec3s pixelX = sx.at<Vec3s>(row, col);
            Vec3s pixelY = sy.at<Vec3s>(row, col);
            Vec3b pixel = Vec3b(0, 0, 0);
            for (int k = 0; k < 3; k++) {
                pixel[k] = sqrt(pow(pixelX[k], 2) + pow(pixelY[k], 2));  // Eucledian distance
            }
            dst.at<Vec3b>(row, col) = pixel;
        }
    }
    return 0;
}

/*  Purpose: Blurs an image using a 5x5 Gaussian kernel and quantizes the colors based on the number of levels
    * @param src: source image
    * @param dst: destination image
    * @param levels: number of colors in the palette
    * @return 0 if successful
*/

int blurQuantize( cv::Mat &src, cv::Mat &dst, int levels ) {
    dst = src.clone();
    blur5x5_2(src, dst);

    float stepSize = 255.0f / levels;

    for (int row=0; row < dst.rows; row++) {
        for (int col = 0; col < dst.cols; col++) {
            Vec3b pixel = dst.at<Vec3b>(row, col);
            for (int k = 0; k < 3; k++) {
                float x = pixel[k];
                // Quantize the color channel value
                int xt = static_cast<int>(x / stepSize);
                int xf = static_cast<int>(xt * stepSize);
                xf = min(xf, 255);

                // Update the color channel value
                pixel[k] = xf;
            }
            dst.at<Vec3b>(row, col) = pixel;
        }
    }
    return 0;
}

/*  Purpose: Quantizes the colors based on the number of levels
    * @param src: source image
    * @param dst: destination image
    * @param levels: number of colors in the palette
    * @return 0 if successful
*/

int Quantize(Mat &src, Mat &dst, int levels) {
    dst = src.clone();

    float stepSize = 255.0f / levels;

    for (int row=0; row < dst.rows; row++) {
        for (int col = 0; col < dst.cols; col++) {
            Vec3b pixel = dst.at<Vec3b>(row, col);
            for (int k = 0; k < 3; k++) {
                float x = pixel[k];
                // Quantize the color channel value
                int xt = static_cast<int>(x / stepSize);
                int xf = static_cast<int>(xt * stepSize);
                xf = min(xf, 255);

                // Update the color channel value
                pixel[k] = xf;
            }
            dst.at<Vec3b>(row, col) = pixel;
        }
    }
    return 0;
}

/*  Purpose: Applies an emboss effect to an image
    * @param src: source image
    * @param dst: destination image
    * @return 0 if successful
*/

int emboss(Mat &src, Mat &dst) {

    dst = src.clone();
    Mat sobelX, sobelY;
    sobelX3x3(src, sobelX);
    sobelY3x3(src, sobelY);

    for(int row = 0; row < src.rows; row++) {
        for(int col = 0; col < src.cols; col++) {
            Vec3b pixel = dst.at<Vec3b>(row, col);
            Vec3s pixelX = sobelX.at<Vec3s>(row, col);
            Vec3s pixelY = sobelY.at<Vec3s>(row, col);

            for(int k = 0; k < 3; k++) {
                float dotProduct = 0.7071 * pixelX[k] + 0.7071 * pixelY[k];
                int embossValue = saturate_cast<uchar>(128 + dotProduct/2);
                pixel[k] = embossValue;
            }
            dst.at<Vec3b>(row, col) = pixel;
        }
    }
    return 0;
}

/*  Purpose: Applies an inverse effect to an image (creates a negative of the image)
    * @param src: source image
    * @param dst: destination image
    * @return 0 if successful
*/

int inverse(Mat &src, Mat &dst) {

    dst = src.clone();
    for(int row = 0; row < src.rows; row++) {
        for(int col = 0; col < src.cols; col++) {
            Vec3b pixel = dst.at<Vec3b>(row, col);
            for(int k = 0; k < 3; k++) {
                pixel[k] = 255 - pixel[k];
            }
            dst.at<Vec3b>(row, col) = pixel;
        }
    }
    return 0;
}

/*  Purpose: Detects faces in an image and paints the rest of the image in greyscale
    * @param src: source image
    * @param dst: destination image
    * @param faces: vector of faces detected
    * @return 0 if successful
*/

int colorFace(Mat &src, Mat &dst, vector<Rect> &faces) {
    dst = src.clone();

    Mat mask = Mat::zeros(src.size(), src.type());
    for(Rect face : faces) {
        rectangle(mask, face, Scalar(255, 255, 255), -1);
    }

    Mat greySrc;
    greyscale(src, greySrc);

    greySrc.copyTo(dst, 255 - mask);
    return 0;
}

/*  Purpose: Applies a painterly effect to an image
    * @param src: source image
    * @param dst: destination image
    * @return 0 if successful
*/

int paintify(Mat &src, Mat &dst) {

    // 1. Color Quantization: Reduce color palette for painterly effect
    dst = src.clone();
    int levels = 8; 
    Quantize(src, dst, 8); // Quantize colors uses quantize function defined above

    // 2. Edge Detection: Extract edges for brushstroke-like outlines
    Mat edges;
    Canny(dst, edges, 50, 150); 

    // 3. Blending: Combine edges with quantized image
    bitwise_and(dst, dst, edges); 
    return 0;
}

/*  Purpose: Applies a texture effect to an image
    * @param src: source image
    * @param dst: destination image
    * @return 0 if successful
*/

int texture(Mat &src, Mat &dst) {
    dst = src.clone();
    Mat texture_image = imread("../../texture.jpeg");
    if(texture_image.empty()) {
        cout << "Error loading texture image" << endl;
        return -1;
    }
    resize(texture_image, texture_image, src.size());

    for(int row = 0; row < src.rows; row++) {
        for(int col = 0; col < src.cols; col++) {
            Vec3b pixel = dst.at<Vec3b>(row, col);
            Vec3b texturePixel = texture_image.at<Vec3b>(row, col);
            for(int k = 0; k < 3; k++) {
                pixel[k] = saturate_cast<uchar>(pixel[k] * texturePixel[k] / 255);
            }
            dst.at<Vec3b>(row, col) = pixel;
        }
    }
    return 0;
}

/*  Purpose: Applies a cartoon effect to an image
    * @param src: source image
    * @param dst: destination image
    * @return 0 if successful
*/

int cartoonize(Mat &src, Mat &dst) {
    dst = src.clone();
    resize(dst, dst, Size(640, 480));
    cvtColor(dst, dst, COLOR_BGR2GRAY);
    medianBlur(dst, dst, 5);
    Mat edges;
    adaptiveThreshold(dst, edges, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 9, 9);
    Mat color;
    bilateralFilter(src, color, 9, 300, 300);
    Mat cartoon;
    bitwise_and(color, color, cartoon, edges);
    dst = cartoon;
    return 0;
}