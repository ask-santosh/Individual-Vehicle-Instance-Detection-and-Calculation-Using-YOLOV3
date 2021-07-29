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

<p align="center">
  <img src="https://github.com/ask-santosh/Individual-Vehicle-Instance-Detection-and-Calculation-Using-YOLOV3/blob/main/Screenshot%20from%202021-07-01%2002-07-38.png">
</p>

## Prerequisites
* Linux distro or MacOS (Tested on Ubuntu 18.04)
* A street video file to run the vehicle counting 
* The pre-trained yolov3 weight file should be downloaded by following these steps:
```
wget https://pjreddie.com/media/files/yolov3.weights
``` 
## Run
### Requirements
* Python3 (Tested on Python 3.6.9
* OpenCV 3.4 or above(Tested on OpenCV 3.4.2.17)
* Imutils
* Scipy
* Numpy

### Usage
* `--input` or `-i` argument requires the path to the input video
* `--output` or `-o` argument requires the path to the output video
* `--yolo` or `-y` argument requires the path to the folder where the configuration file, weights and the coco.names file is stored
* `--confidence` or `-c` is an optional argument which requires a float number between 0 to 1 denoting the minimum confidence of detections. By default, the confidence is 0.5 (50%).
* `--threshold` or `-t` is an optional argument which requires a float number between 0 to 1 denoting the threshold when applying non-maxima suppression. By default, the threshold is 0.3 (30%).
* `--use-gpu` or `-u` is an optional argument which requires 0 or 1 denoting the use of GPU. By default, the CPU is used for computations
Examples: 
* Running with defaults
```
python3 yolo_video.py --input asia.mp4 --output asia_new.mp4 --yolo yolo-coco 
```
* Specifying confidence (0 < confidence <= 1)
```
python3 yolo_video.py --input asia.mp4 --output asia_new.mp4 --yolo yolo-coco --confidence 0.3
```
* Using GPU(For that one should enabled CUDA & CUDNN)
```
python3 yolo_video.py --input asia.mp4 --output asia_new.mp4 --yolo yolo-coco --use-gpu 1
```

## Detection of objects and ID assignment
* The detections are performed on each frame by using YOLOv3(288) object detection algorithm and displayed on the screen with bounding boxes.
* The detections are filtered to keep all vehicles like motorcycle, bus, car, cycle, truck, train. The reason why trains are also counted is because sometimes, the longer vehicles like a bus, is detected as a train; therefore, the trains are also taken into account(just for demonstration purpose).
* The center of each box is taken as a reference point (denoted by a green dot when performing the detections) when track the vehicles.   
* Also, in order to track the vehicles, the shortest distance to the center point is calculated for each vehicle in the last 10 frames. 
* If `shortest distance < max(width, height) / 2`, then the vehicles is not counted in the current frame. Else, the vehicle is counted again. Usually, the direction in which the vehicle moves is bigger than the other one. 
* For example, if a vehicle moves from North to South or South to North, the height of the vehicle is most likely going to be greater than or equal to the width. Therefore, in this case, `height/2` is compared to the shortest distance in the last 10 frames. 
* As YOLO misses a few detections for a few consecutive frames, this issue can be resolved by saving the detections for the last 10 frames and comparing them to the current frame detections when required. The size of the vehicle does not vary too much in 10 frames and has been tested in multiple scenarios; therefore, 10 frames was chosen as an optimal value.
* All the detected classes are stored in a dictionary for further checking whether the same id is repeated or not and from that counter is executed for further calculations .
