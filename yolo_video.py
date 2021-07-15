# import the necessary packages
import numpy as np
import imutils
import time
from scipy import spatial
import cv2
from input_retrieval import *
import os

# All these classes will be counted as 'vehicles'
list_of_vehicles = ["bicycle", "car", "motorbike", "bus", "truck", "train"]
final_dict = {}
# Setting the threshold for the number of frames to search a vehicle for
FRAMES_BEFORE_CURRENT = 10
inputWidth, inputHeight = 416, 416

# Parse command line arguments and extract the values required
LABELS, weightsPath, configPath, inputVideoPath, outputVideoPath, \
preDefinedConfidence, preDefinedThreshold, USE_GPU = parseCommandLineArguments()

# Initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                           dtype="uint8")


# PURPOSE: Displaying the FPS of the detected video
# PARAMETERS: Start time of the frame, number of frames within the same second
# RETURN: New start time, new number of frames 
def displayFPS(start_time, num_frames):
    current_time = int(time.time())
    if (current_time > start_time):
        os.system('clear')  # Equivalent of CTRL+L on the terminal
        print("FPS:", num_frames)
        num_frames = 0
        start_time = current_time
    return start_time, num_frames


# PURPOSE: Draw all the detection boxes with a green dot at the center

def drawDetectionBoxes(idxs, boxes, classIDs, confidences, frame):
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indices we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the frame
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                                       confidences[i])
            cv2.putText(frame, text, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            # Draw a green dot in the middle of the box
            cv2.circle(frame, (x + (w // 2), y + (h // 2)), 2, (0, 0xFF, 0), thickness=2)


# PURPOSE: Initializing the video writer with the output video path and the same number
# of fps, width and height as the source video
# PARAMETERS: Width of the source video, Height of the source video, the video stream
# RETURN: The initialized video writer

def initializeVideoWriter(video_width, video_height, videoStream):
    # Getting the fps of the source video
    sourceVideofps = videoStream.get(cv2.CAP_PROP_FPS)
    # initialize our video writer
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    return cv2.VideoWriter(outputVideoPath, fourcc, sourceVideofps,
                           (video_width, video_height), True)


# PURPOSE: Identifying if the current box was present in the previous frames
# PARAMETERS: All the vehicular detections of the previous frames, 
#			the coordinates of the box of previous detections
# RETURN: True if the box was current box was present in the previous frames;
#		  False if the box was not present in the previous frames

def boxInPreviousFrames(previous_frame_detections, current_box, current_detections):
    centerX, centerY, width, height = current_box
    dist = np.inf  # Initializing the minimum distance
    # Iterating through all the k-dimensional trees
    for i in range(FRAMES_BEFORE_CURRENT):
        coordinate_list = list(previous_frame_detections[i].keys())
        if len(coordinate_list) == 0:  # When there are no detections in the previous frame
            continue
        # Finding the distance to the closest point and the index
        temp_dist, index = spatial.KDTree(coordinate_list).query([(centerX, centerY)])
        if (temp_dist < dist):
            dist = temp_dist
            frame_num = i
            coord = coordinate_list[index[0]]

    if (dist > (max(width, height) / 2)):
        return False

    # Keeping the vehicle ID constant
    current_detections[(centerX, centerY)] = previous_frame_detections[frame_num][coord]
    return True


def count_vehicles(idxs, boxes, classIDs, vehicle_count, previous_frame_detections, frame):
    current_detections = {}
    id_mapper = {}  # for storing all ids and classes that are detected in frame
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indices we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            centerX = x + (w // 2)
            centerY = y + (h // 2)

            # When the detection is in the list of vehicles, AND
            # the ID of the detection is not present in the vehicles
            if (LABELS[classIDs[i]] in list_of_vehicles):
                current_detections[(centerX, centerY)] = vehicle_count
                id_mapper[vehicle_count] = LABELS[classIDs[i]]
                if (not boxInPreviousFrames(previous_frame_detections, (centerX, centerY, w, h), current_detections)):
                    vehicle_count += 1

                ID = current_detections.get((centerX, centerY))
                # If there are two detections having the same ID due to being too close,
                # then assign a new ID to current detection.
                if (list(current_detections.values()).count(ID) > 1):
                    current_detections[(centerX, centerY)] = vehicle_count
                    id_mapper[vehicle_count] = LABELS[classIDs[i]]
                    vehicle_count += 1

                # Display the ID at the center of the box
                cv2.putText(frame, str(ID), (centerX, centerY), \
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 255], 2)

    global final_dict
    final_dict.update(id_mapper)
    return vehicle_count, current_detections, id_mapper


# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# Using GPU if flag is passed
if USE_GPU:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
videoStream = cv2.VideoCapture(inputVideoPath)
video_width = int(videoStream.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(videoStream.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialization
previous_frame_detections = [{(0, 0): 0} for i in range(FRAMES_BEFORE_CURRENT)]
# previous_frame_detections = [spatial.KDTree([(0,0)])]*FRAMES_BEFORE_CURRENT # Initializing all trees
num_frames, vehicle_count = 0, 0
writer = initializeVideoWriter(video_width, video_height, videoStream)
start_time = int(time.time())

# loop over frames from the video file stream
while True:
    num_frames += 1
    # Initialization for each iteration
    boxes, confidences, classIDs = [], [], []
    vehicle_crossed_line_flag = False

    # Calculating fps each second
    start_time, num_frames = displayFPS(start_time, num_frames)
    # read the next frame from the file
    (grabbed, frame) = videoStream.read()

    # if the frame was not grabbed, then we have reached the end of the stream
    if not grabbed:
        break

    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (inputWidth, inputHeight),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for i, detection in enumerate(output):
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > preDefinedConfidence:
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and height
                box = detection[0:4] * np.array([video_width, video_height, video_width, video_height])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, preDefinedConfidence,
                            preDefinedThreshold)

    # Draw detection box
    drawDetectionBoxes(idxs, boxes, classIDs, confidences, frame)

    vehicle_count, current_detections, id_mapper = count_vehicles(idxs,
                                                                  boxes,
                                                                  classIDs,
                                                                  vehicle_count,
                                                                  previous_frame_detections,
                                                                  frame)

    # write the output frame to disk
    writer.write(frame)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Updating with the current frame detections
    previous_frame_detections.pop(0)  # Removing the first frame from the list
    # previous_frame_detections.append(spatial.KDTree(current_detections))
    previous_frame_detections.append(current_detections)

# Calculating individual instances of each class
print("final dictionary=========>>", final_dict)
count_car = 0
count_motorbike = 0
count_truck = 0
count_bicycle = 0
count_bus = 0
count_train = 0
for k, v in final_dict.items():
    if v == "car":
        count_car += 1
    elif v == "motorbike":
        count_motorbike += 1
    elif v == "truck":
        count_truck += 1
    elif v == "bus":
        count_bus += 1
    elif v == "bicycle":
        count_bicycle += 1
    else:
        count_train += 1

print("no_cars==========", count_car)
print("no_motorbike==========", count_motorbike)
print("no_truck==========", count_truck)
print("no_bus==========", count_bus)
print("no_bicycle==========", count_bicycle)
print("no_train==========", count_train)

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
videoStream.release()
