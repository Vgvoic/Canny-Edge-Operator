﻿# Canny-Edge-Operator

 This is the Canny Edge Operator manually written using knowledge gained through the subjects I've been attending on the Faculty of Engineering in Rijeka.
 My interpretation of the Canny Edge Operator (In text reffered to as CEO) using sources from around the internet related to the various steps that the CEO is composed of.

CEO is broken down into several steps:
  - Gaussian Filter
  - Grayscale conversion
  - Intensity Gradients
  - Gradient magnitude detection (done through the Sobel Gradient)
  - Non Maximum Supression
  - Double Threshold
  - Hysteresis.

The P6 format, aside from it being used because of the nature of the assignment, allows manipulation of the image on a pixel level; the image is written as a collection of structures containing the x and y positions of the pixel in question, as well as its respective RGB values.


The assignment has been graded with a B in the Bologne Process, and has been marked as a success. THe code doesn't entirely correspond with the expected output of the CEO, but has been a valuable lesson regarding image manipulation and bare-bone pixel matrix transformations.
