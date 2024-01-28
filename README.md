# Image-and-Video-Manipulation

CS5330 - Pattern Recognition and Computer Vision <br>
Assignment 1 <br>
Image Manipulation Program <br>

Contributors <br>
Name : Haritha Selvakumaran <br>

Video Link <br>
https://drive.google.com/file/d/1SbZzqYNAO63MQ8H_pXS8nA335XAjZN2g/view?usp=sharing <br>

Environment <br>
Operating System: Windows 11 Home x64 <br>
IDE : Visual Studio Code <br>

Overview: <br>
This project is about developing an image and video processing and manipulation application using OpenCV along with C++ libraries. The program constantly captures the video in the form of stream and works on each frame based on the key pressed by the user. It primarily has a few features such as greyscale conversion, sepia filter, Sobel filters, detection of face, etc. There are also a few additional features where the user can choose cartoony filter, paintify filters, texture synthesis filters and embossing effect. In addition, the user can also save the images or videos and as well as convert the video back to its original form. The filter.cpp and filter.h contain all the manipulation functions, while the vidDisplay.cpp is the main file that checks the key pressed and calls the corresponding filters accordingly. The following key press are used for these corresponding function:

g - Grayscale <br>
h - Alternate Grayscale <br>
p - Sepia <br>
x - Sobel X <br>
y - Sobel Y <br>
m - Magnitude <br>
b - Blur <br>
l - Quantization <br>
f - Detect faces <br>
e - Emboss <br>
c - Inverse Colors <br>
d - Converting anything other than the face to grayscale <br>
z - Paintify <br>
t - Texture <br>
a - Cartoonify <br>
n - Remove all filters <br>
s - Save the image with filters <br>
v - Save a short video with filters <br>
q - Quit the program
