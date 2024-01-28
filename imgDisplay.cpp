// Haritha Selvakumaran 01/24/2024
// Purpose: Display image in a window

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <stdio.h>

using namespace cv;
using namespace std;

// The main function will read an image and display it in a window

int main()
{
    string filepath = "C:/Users/harit/imgDisplay/data/bird.jpeg";
    Mat image;
    image = imread(filepath);

    if (image.empty())
    {
        cout << ("No image data") << endl;
        return -1;
    }

    namedWindow("Image Display", WINDOW_AUTOSIZE);
    imshow("Image Display", image);

    while (true)
    {
        char k = waitKey(1);
        if (k == 'q')
        {
            cout << ("Quitting") << endl;
            break;
        }
        else if (k == 's')
        {
            imwrite("C:/Users/harit/imgDisplay/data/bird_copy.jpeg", image);
            cout << ("Image saved") << endl;
        }
    }
    destroyWindow("Image Display");
    return 0;
}
