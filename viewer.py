#ipcam.py
import sys
#sys.path.append(r'C:\Users\dledb\Anaconda3\Lib\site-packages')
#sys.path.append(r'C:\darknet\build\darknet\x64')
# opencv object tracking
# object detection and tracking opencv
import cv2
import numpy as np
# Load Yolo

"""

 update the below yolo weight and config files to the location of the files on your computer.
 After installing darknet it is usually in a "darknet" folder  - David

"""
yolo_weight = r"C:\darknet\build\darknet\x64\yolov4.weights"
yolo_config = r"C:\darknet\build\darknet\x64\cfg\yolov4.cfg"
coco_labels = r"C:\darknet\build\darknet\x64\data\coco.names"
net = cv2.dnn.readNet(yolo_weight, yolo_config)
classes = []
with open(coco_labels, "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
 

# network, class_names, class_colors = load_network("cfg/yolov4-csp.cfg", "cfg/coco.data", "yolov4-csp.weights") 
 
# Below function will read video frames
cap = cv2.VideoCapture(0)
boundingbox = ''
count = 0
while True:
    read_ok, img = cap.read()
    height, width, channels = img.shape
    # boundingboxarray = np.zeros([height, width, 4], dtype=np.uint8)

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    outs = net.forward(output_layers)
    
    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []

    print("\n","Start Outs")
    for out in outs:
        print(out.shape)

        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]            
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    print("End Outs", "\n")

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
 
    font = cv2.FONT_HERSHEY_DUPLEX
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence_label = int(confidences[i] * 100)
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, f'{label, confidence_label}', (x-25, y + 75), font, 2, color, 2)
 
    cv2.imshow("Image", img)
    # Close video window by pressing 'x'
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break
