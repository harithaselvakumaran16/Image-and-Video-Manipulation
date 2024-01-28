// Haritha Selvakumaran 01/23/2020
// Main file for vidDisplay that performs various image manipulation functions

// Importing libraries

#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include "filter.h"
#include "faceDetect.h"

using namespace cv;
using namespace std;

// Main function

/* Here we take in the video from the webcam and perform various image manipulation functions based on
 * the key pressed by the user. We identify the key press using the waitKey() function and perform the
 * corresponding function.
 */

int main(int argc, char *argv[])
{
        VideoCapture *capdev;

        capdev = new VideoCapture(0);
        if (!capdev->isOpened())
        {
                printf("Unable to open video device\n");
                return (-1);
        }

        // Get some properties of the frame
        Size refS((int)capdev->get(CAP_PROP_FRAME_WIDTH),
                  (int)capdev->get(CAP_PROP_FRAME_HEIGHT));
        printf("Expected size: %d %d\n", refS.width, refS.height);

        namedWindow("Video", 1);
        Mat frame;
        Mat processed_frame;
        bool isGreyscale = false;
        bool isCustomGreyscale = false;
        bool isSepia = false;
        bool isSobelX = false;
        bool isSobelY = false;
        bool isMagnitude = false;
        bool isBlur = false;
        bool isQuantize = false;
        bool isFaceDetect = false;
        bool isEmboss = false;
        bool isInverse = false;
        bool isColorFace = false;
        bool isPaintify = false;
        bool isTexture = false;
        bool isCartoonize = false;
        Mat sobelX, sobelY, mag, greyscale_frame, embossing_frame;
        vector<Rect> faces;
        int count = 0;

        while (true)
        {
                *capdev >> frame; // Get a new frame from the camera as a stream
                if (frame.empty())
                {
                        printf("Frame is empty\n");
                        break;
                }
                if (isGreyscale)
                {
                        cvtColor(frame, processed_frame, COLOR_BGR2GRAY);
                }
                else if (isCustomGreyscale)
                {
                        greyscale(frame, processed_frame);
                }
                else if (isSepia)
                {
                        sepia(frame, processed_frame);
                }
                else if (isSobelX)
                {
                        sobelX3x3(frame, sobelX);
                        convertScaleAbs(sobelX, processed_frame);
                }
                else if (isSobelY)
                {
                        sobelY3x3(frame, sobelY);
                        convertScaleAbs(sobelY, processed_frame);
                }
                else if (isMagnitude)
                {
                        sobelX3x3(frame, sobelX);
                        isSobelX = false;
                        sobelY3x3(frame, sobelY);
                        isSobelY = false;
                        magnitude(sobelX, sobelY, mag);
                        convertScaleAbs(mag, processed_frame);
                }
                else if (isBlur)
                {
                        blur5x5_2(frame, processed_frame);
                }
                else if (isQuantize)
                {
                        blurQuantize(frame, processed_frame, 10);
                }
                else if (isFaceDetect)
                {
                        cvtColor(frame, greyscale_frame, COLOR_BGR2GRAY);
                        detectFaces(greyscale_frame, faces);
                        drawBoxes(frame, faces, 50, 1.0);
                }
                else if (isColorFace)
                {
                        cvtColor(frame, greyscale_frame, COLOR_BGR2GRAY);
                        detectFaces(greyscale_frame, faces);
                        drawBoxes(frame, faces, 50, 1.0);
                        colorFace(frame, processed_frame, faces);
                }
                else if (isEmboss)
                {
                        emboss(frame, processed_frame);
                }
                else if (isInverse)
                {
                        inverse(frame, processed_frame);
                }
                else if (isPaintify)
                {
                        paintify(frame, processed_frame);
                }
                else if (isTexture)
                {
                        texture(frame, processed_frame);
                }
                else if (isCartoonize)
                {
                        cartoonize(frame, processed_frame);
                }
                else
                {
                        processed_frame = frame;
                }

                // Wait for keypress
                int key = waitKey(10);

                if (key != -1)
                {
                        // Reset all filter flags
                        isGreyscale = false;
                        isCustomGreyscale = false;
                        isSepia = false;
                        isSobelX = false;
                        isSobelY = false;
                        isMagnitude = false;
                        isBlur = false;
                        isQuantize = false;
                        isFaceDetect = false;
                        isEmboss = false;
                        isInverse = false;
                        isColorFace = false;
                        isPaintify = false;
                        isTexture = false;
                        isCartoonize = false;

                        if (key == 'q')
                        {
                                cout << "Quitting" << endl;
                                break;
                        }
                        if (key == 's')
                        {
                                count++;
                                cout << "Saving image" << endl;

                                bool isSaved = imwrite("C:/Users/harit/vidDisplay/data/image_" + to_string(count) + ".jpg", processed_frame);
                                if (isSaved)
                                {
                                        cout << "Image saved successfully" << endl;
                                }
                                else
                                {
                                        cout << "Failed to save image" << endl;
                                }
                        }

                        VideoWriter video;
                        if (key == 'v')
                        {
                                cout << "Saving video" << endl;

                                video.open("C:/Users/harit/vidDisplay/data/video.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, Size(refS.width, refS.height), true);
                                if (video.isOpened())
                                {
                                        cout << "Video saved successfully" << endl;
                                }
                                else
                                {
                                        cout << "Failed to save video" << endl;
                                }
                        }

                        else if (key == 'g')
                        {
                                isGreyscale = true;
                                cout << "Converting to greyscale" << endl;
                        }
                        else if (key == 'h')
                        {
                                isCustomGreyscale = true;
                                cout << "Alternate Greyscale transformation" << endl;
                        }
                        else if (key == 'p')
                        {
                                isSepia = true;
                                cout << "Converting to sepia" << endl;
                        }
                        else if (key == 'x')
                        {
                                isSobelX = true;
                                cout << "Converting to sobel X" << endl;
                        }
                        else if (key == 'y')
                        {
                                isSobelY = true;
                                cout << "Converting to sobel Y" << endl;
                        }
                        else if (key == 'm')
                        {
                                isMagnitude = true;
                                cout << "Converting to magnitude" << endl;
                        }
                        else if (key == 'b')
                        {
                                isBlur = true;
                                cout << "Converting to blur1" << endl;
                        }
                        else if (key == 'l')
                        {
                                isQuantize = true;
                                cout << "Converting to blur using quantization" << endl;
                        }
                        else if (key == 'f')
                        {
                                isFaceDetect = true;
                                cout << "Detecting faces" << endl;
                        }
                        else if (key == 'e')
                        {
                                isEmboss = true;
                                cout << "Converting to emboss" << endl;
                        }
                        else if (key == 'c')
                        {
                                isInverse = true;
                                cout << "Converting to inverse colours" << endl;
                        }
                        else if (key == 'd')
                        {
                                isColorFace = true;
                                cout << "Converting to color face" << endl;
                        }
                        else if (key == 'z')
                        {
                                isPaintify = true;
                                cout << "Converting to paintify" << endl;
                        }
                        else if (key == 't')
                        {
                                isTexture = true;
                                cout << "Converting to texture" << endl;
                        }
                        else if (key == 'a')
                        {
                                isCartoonize = true;
                                cout << "Converting to cartoonize" << endl;
                        }
                        if (key == 'n')
                        {
                                isGreyscale = false;
                                isCustomGreyscale = false;
                                isSepia = false;
                                isSobelX = false;
                                isSobelY = false;
                                isMagnitude = false;
                                isBlur = false;
                                isQuantize = false;
                                isFaceDetect = false;
                                isEmboss = false;
                                cout << "Resetting to original" << endl;
                        }
                }

                        imshow("Video", processed_frame); // Display processed frame
        }
        delete capdev;
        return (0);
}
