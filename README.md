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
This project encapsulates the development of a robust image and video processing application utilizing OpenCV and C++ libraries. Operating seamlessly in real-time, the program continuously captures video streams and dynamically processes each frame based on user input. Its core functionalities include standard features like greyscale conversion, sepia filters, Sobel filters, and face detection. Additionally, users can choose from an array of creative filters such as cartoony, paintify, texture synthesis, and embossing effects.

The modular structure of the project is organized around filter.cpp and filter.h, housing all manipulation functions, while vidDisplay.cpp serves as the main control hub. The latter detects user key inputs and orchestrates the application of corresponding filters. A notable feature is the ability to save modified images or videos, providing users with the flexibility to revert the video back to its original state.

This project not only showcases technical prowess in image processing but also prioritizes user interaction and creative expression. The interplay of efficient algorithms and an intuitive user interface establishes a versatile tool for visual content manipulation. The combination of real-time processing, diverse filters, and data preservation features makes this application a comprehensive solution for image and video enhancement. The following key press are used for these corresponding function:

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
