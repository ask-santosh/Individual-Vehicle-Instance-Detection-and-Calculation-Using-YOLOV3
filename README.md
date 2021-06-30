# Individual-Vehicle-Instance-Detection-and-Calculation-Using-YOLOV3
## Overview
You Only Look Once (YOLO) is a CNN architecture for performing real-time object detection. The algorithm applies a single neural network to the full video, and then divides the video into regions and predicts bounding boxes and probabilities for each region. For more detailed working of YOLO algorithm, please refer to the [YOLO paper](https://pjreddie.com/media/files/papers/YOLOv3.pdf). 

This project aims to count every vehicle (motorcycle, bus, car, cycle, truck, train) detected in the input video using YOLOv3 object-detection algorithm.

## Working 
<p align="center">
  <img src="https://github.com/ask-santosh/Individual-Vehicle-Instance-Detection-and-Calculation-Using-YOLOV3/blob/main/asia_new.gif">
</p>

As shown in the image above, when the vehicles in the frame are detected, they are stored in a dictionary with their id. After getting detected once, the vehicles get tracked and do not get re-counted by the algorithm. 

You may also notice that the vehicles will initially be detected and the individual instance counter increments, but for a few frames, the vehicle is not detected, and then it gets detected again. As the vehicles are tracked, the vehicles are not re-counted if they are counted once. 

Finally at the end of the video all the individual instances are calculated and printed on the  terminal .

## Prerequisites
* Linux distro or MacOS (Tested on Ubuntu 18.04)
* A street video file to run the vehicle counting 
* The pre-trained yolov3 weight file should be downloaded by following these steps:
```
wget https://pjreddie.com/media/files/yolov3.weights
``` 
