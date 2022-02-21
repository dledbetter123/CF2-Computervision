#ipcam.py
import sys
import socket
import numpy as np

sys.path.append(r'C:\Users\dledb\Anaconda3\Lib\site-packages')

import cv2

###
# Configuration Step for darknet in python.
###
yolo_weight = r"C:\darknet\build\darknet\x64\yolov4.weights"
yolo_config = r"C:\darknet\build\darknet\x64\cfg\yolov4.cfg"
coco_labels = r"C:\darknet\build\darknet\x64\data\coco.names"
net = cv2.dnn.readNet(yolo_weight, yolo_config)

# Load coco object names file
classes = []
with open(coco_labels, "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

yolov4_width = 640
yolov4_height = 480

###
# End Darknet config
###

client_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
host_ip = '192.168.4.1' # paste your server ip address here
port = 5000
print("Connecting to socket on {}:{}...".format(host_ip, port))
client_socket.connect((host_ip,port)) # a tuple
print("Connection Successful")

imgdata = None
data_buffer = b''
# cap = cv2.VideoCapture('rtsp://admin:123456@192.168.4.1/H264?ch=1&subtype=0')
# cap = cv2.VideoCapture(0)

while True:
    data_buffer+=(client_socket.recv(512))
    start_idx = data_buffer.find(b'\xff\xd8')
    end_idx = data_buffer.find(b'\xff\xd9')
    # print("found begin: ", a, " found end: ", b)
    i = None

    # At startup we might get an end before we get the first start, if
    # that is the case then throw away the data before start
    if end_idx > -1 and end_idx < start_idx:
        data_buffer = data_buffer[start_idx:]

    # We have a start and an end of the image in the buffer now
    if start_idx > -1 and end_idx > -1 and end_idx > start_idx:
        # Pick out the image to render ...
        imgdata = data_buffer[start_idx:end_idx + 2]
        # .. and remove it from the buffer
        data_buffer = data_buffer[end_idx + 2 :]
        print(imgdata)
        i = cv2.imdecode(np.fromstring(imgdata, dtype=np.uint8),cv2.IMREAD_GRAYSCALE)

        #print('About to start the Read command')
        img = i
        
        # gives frame 3 channels
        img = np.stack((img,)*3, axis=-1)
    
        height, width, channels = img.shape
    
        # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)
    
        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
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

# cap.release()
client_socket.close()
cv2.destroyAllWindows()