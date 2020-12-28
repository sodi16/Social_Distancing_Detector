import numpy as np
from tensorflow.keras.models import load_model
import math
import cv2
from PIL import Image, ImageDraw
import os
import io
import socket


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.objness = objness
        self.classes = classes
        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]

        return self.score


def _sigmoid(x):
    return 1. / (1. + np.exp(-x))


def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    boxes = []
    netout[..., :2] = _sigmoid(netout[..., :2])
    netout[..., 4:] = _sigmoid(netout[..., 4:])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * netout[..., 5:]
    netout[..., 5:] *= netout[..., 5:] > obj_thresh

    for i in range(grid_h * grid_w):
        row = i / grid_w
        col = i % grid_w
        for b in range(nb_box):
            # 4th element is objectness score
            objectness = netout[int(row)][int(col)][b][4]
            if (objectness.all() <= obj_thresh): continue
            # first 4 elements are x, y, w, and h
            x, y, w, h = netout[int(row)][int(col)][b][:4]
            x = (col + x) / grid_w  # center position, unit: image width
            y = (row + y) / grid_h  # center position, unit: image height
            w = anchors[2 * b + 0] * np.exp(w) / net_w  # unit: image width
            h = anchors[2 * b + 1] * np.exp(h) / net_h  # unit: image height
            # last elements are class probabilities
            classes = netout[int(row)][col][b][5:]
            box = BoundBox(x - w / 2, y - h / 2, x + w / 2, y + h / 2, objectness, classes)
            boxes.append(box)
    return boxes


def is_close2(x1, y1, x2, y2):
    dst = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return dst


def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    new_w, new_h = net_w, net_h
    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w) / 2. / net_w, float(new_w) / net_w
        y_offset, y_scale = (net_h - new_h) / 2. / net_h, float(new_h) / net_h
        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)
    return boxes


def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b
    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3


def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
    intersect = intersect_w * intersect_h
    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin
    union = w1 * h1 + w2 * h2 - intersect
    return float(intersect) / union


def do_nms(boxes, nms_thresh):
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return
    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])
        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            if boxes[index_i].classes[c] == 0: continue
            for j in range(i + 1, len(sorted_indices)):
                index_j = sorted_indices[j]
                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0
    return boxes


def load_image_pixels(image_array, input_w, input_h):
    image_array = cv2.resize(image_array, (input_w, input_h))
    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array.astype('float32')
    image_array /= 255.0
    image_h, image_w = image_array.shape[1:3]
    return image_array


# get all of the results above a threshold
def get_boxes(boxes, labels, thresh):
    v_boxes, v_labels, v_scores = list(), list(), list()
    # enumerate all boxes
    for box in boxes:
        # enumerate all possible labels
        for i in range(len(labels)):
            # check if the threshold for this label is high enough
            if box.classes[i] > thresh:
                v_boxes.append(box)
                v_labels.append(labels[i])
                v_scores.append(box.classes[i] * 100)
            # don't break, many labels may trigger for one box
    return v_boxes, v_labels, v_scores


def centroid_box(labels, boxes):
    array_centroid = {}
    for i in range(len(boxes)):
        if labels[i] == 'person':
            box = boxes[i]
            y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
            width, height = x2 - x1, y2 - y1
            centroid_x = x1 + int(width / 2)
            centroid_y = y1 + int(height / 2)
            array_centroid[i] = [centroid_y, centroid_x]
    return array_centroid


def find_close_persons(centroid):
    close_people = []
    safe_people = []

    for i in centroid.items():
        for y in centroid.items():
            if i[0] != y[0]:
                dist = is_close2(i[1][0], i[1][1], y[1][0], y[1][1])
                if dist < 77:
                    if i[0] not in close_people:
                        close_people.append(i[0])
                    if y[0] not in close_people:
                        close_people.append(y[0])
        if i[0] not in close_people:
            safe_people.append(i[0])
    return close_people, safe_people


def draw_boxes2(data, v_boxes, v_centroid, red_area, green_area):
    rect = None
    for i in v_centroid.keys():
        box = v_boxes[i]
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        width, height = x2 - x1, y2 - y1
        if i in red_area:
            # rect = Rectangle((x1, y1), width, height, fill=False, color='red', lw=2)
            rect = cv2.rectangle(data, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
        elif i in green_area:
            # rect = Rectangle((x1, y1), width, height, fill=False, color='green', lw=2)
            rect = cv2.rectangle(data, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
    return rect


def draw_boxes3(rect, v_boxes, v_centroid, red_area, green_area):
    rect = Image.fromarray((rect * 255).astype(np.uint8))
    for i in v_centroid.keys():
        box = v_boxes[i]
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        width, height = x2 - x1, y2 - y1
        rect_box = ImageDraw.Draw(rect)
        if i in red_area:
            rect_box.rectangle([x1, y1, x2, y2], outline ="red", width=2)
        elif i in green_area:
            rect_box.rectangle([x1, y1, x2, y2], outline ="green", width=2)
    return rect



labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
          "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
          "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
          "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
          "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
          "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
          "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
          "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
          "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
          "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "truck"]

"""labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
          "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
          "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
          "backpack",  "handbag", "tie", "suitcase", "bottle", "wine glass", "cup",
          "chair", "sofa", "pottedplant", "bed", "toilet", "tvmonitor", "laptop", "mouse",
          "remote", "keyboard", "cell phone"]"""


def wait_for_acknowledge(client, response):
    amount_received = 0
    amount_expected = len(response)

    msg = str()
    while amount_received < amount_expected:
        data = client.recv(2000)
        amount_received += len(data)
        msg += data.decode("utf-8")
    return msg


HOST = ''
PORT = 8485
model = load_model('model.h5', compile=False)
input_w, input_h = 416, 416
buff_size = 1024

# initiate connection
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_addr = (socket.gethostname(), 2022)  # change here for sending to another machine in LAN
s.bind(server_addr)
s.listen(15)

client, address = s.accept()
print(f"Connection from {address} has been established!")
###################################################################

cmd_from_client = wait_for_acknowledge(client, "Start sending image.")


# send an ACK
imgCount_from_server = 0
if cmd_from_client == "Start sending image.":

    print("Command \"Start sending image.\" received.")
    client.sendall(bytes("ACK", "utf-8"))
    try:
        print("Client is now waiting for the number of images.")
        imgCount_from_server = int(wait_for_acknowledge(client, str(3)))

    except:
        raise ValueError("Number of images received is buggy.")

if imgCount_from_server > 0:
    print("Number of images to receive: ", imgCount_from_server)
    print("Sending ACK...")
    client.sendall(bytes("ACK", "utf-8"))

print(f"Client is now receiving {imgCount_from_server} images.")

for i in range(imgCount_from_server):
    imgsize = int(wait_for_acknowledge(client, str(3)))
    print(f"\tImage size of {imgsize}B received by Client")
    #sending Ack
    client.sendall(bytes("ACK", "utf-8"))
    buff = client.recv(imgsize)
    image_buff = Image.open(io.BytesIO(buff))
    image_array = np.array(image_buff)

    #load correct format and shape of an image
    image_array = load_image_pixels(image_array, input_w, input_h)

    #making prediction with yolov3
    yhat = model.predict(image_array)
    anchors = [[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]
    class_threshold = 0.6
    boxes = list()

    for lab in range(len(yhat)):
        boxes += decode_netout(yhat[lab][0], anchors[lab], class_threshold, input_h, input_w)
    boxes = correct_yolo_boxes(boxes, image_array.shape[1], image_array.shape[2], input_h, input_w)

    boxes = do_nms(boxes, 0.5)
    v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)
    # summarize what we found
    for box in range(len(v_boxes)):
        print(v_labels[box], v_scores[box])


    v_centroid = centroid_box(v_labels, v_boxes)
    red_area, green_area = find_close_persons(v_centroid)
    image_array = image_array.reshape(image_array.shape[1:])
    rect = draw_boxes3(image_array, v_boxes, v_centroid, red_area, green_area)


    imgByteArr = io.BytesIO()
    rect.save(imgByteArr, format='PNG')
    imgByteArr = imgByteArr.getvalue()

    client.sendall(bytes(str(len(imgByteArr)), "utf-8"))
    # print("Client is now waiting for acknowledge from server.")
    ack_from_client = wait_for_acknowledge(client, "ACK")
    if ack_from_client != "ACK":
        raise ValueError('Client does not acknowledge img size.')

    client.sendall(imgByteArr)

    #rect.show()
    #rect.save(fr'C:\Users\so16s\AppData\Roaming\JetBrains\PyCharmCE2020.3\scratches\YOLO\darknet\new_pic\frame{i}.jpg')


print("All images sent.\nClosing connection.")
client.close()

