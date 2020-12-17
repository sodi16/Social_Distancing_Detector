
# load yolov3 model and perform object detection
# based on https://github.com/experiencor/keras-yolo3
import numpy as np
from numpy import expand_dims
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import math
import pylab
from IPython.display import clear_output
import cv2
import shutil
import os
import io
import socket
import struct
import time
import pickle
import zlib



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
    nb_class = netout.shape[-1] - 5
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


# load and prepare an image
def load_image_pixels(filename, shape):
    image = load_img(filename)
    width, height = image.size
    # load the image with the required size
    image = load_img(filename, target_size=shape)
    # convert to numpy array
    image = img_to_array(image)
    # scale pixel values to [0, 1]
    image = image.astype('float32')
    image /= 255.0
    # add a dimension so that we have one sample
    image = expand_dims(image, 0)
    return image, width, height


def load_frames_of_video(video_path, new_frame_path):
    vidcap = cv2.VideoCapture(video_path)
    count = 0
    if os.path.exists(new_frame_path):
        shutil.rmtree(new_frame_path)
        os.mkdir(new_frame_path)
    else:
        os.mkdir(new_frame_path)
    while vidcap.isOpened():
        ret, image = vidcap.read()
        if ret:
            cv2.imwrite(new_frame_path + r'/frame_{0}.jpg'.format(count), image)
            count += 1
        else:
            vidcap.release()
            cv2.destroyAllWindows()
            break


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

    #tuple (0, [350, 622]) num id centroid, and list centroid position
    for i in centroid.items():
        for y in centroid.items():
            if i[0] != y[0]:
                dist = is_close2(i[1][0], i[1][1], y[1][0], y[1][1])
                if dist < 75:
                    if i[0] not in close_people:
                        close_people.append(i[0])
                    if y[0] not in close_people:
                        close_people.append(y[0])
        if i[0] not in close_people:
            safe_people.append(i[0])
    return close_people, safe_people


# draw all results
def draw_boxes(filename, v_boxes, v_centroid, red_area, green_area):
    data = plt.imread(filename)
    plt.imshow(data)
    ax = plt.gca()
    rect = None
    for i in v_centroid.keys():
        box = v_boxes[i]
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        width, height = x2 - x1, y2 - y1
        if i in red_area:
            rect = Rectangle((x1, y1), width, height, fill=False, color='red', lw=2)
        elif i in green_area:
            rect = Rectangle((x1, y1), width, height, fill=False, color='green', lw=2)
        # draw the box
        ax.add_patch(rect)
    plt.show()

def draw_boxes2(filename, v_boxes, v_centroid, red_area, green_area):
    data = cv2.imread(filename)
    rect = None
    for i in v_centroid.keys():
        box = v_boxes[i]
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        width, height = x2 - x1, y2 - y1
        if i in red_area:
            #rect = Rectangle((x1, y1), width, height, fill=False, color='red', lw=2)
            rect = cv2.rectangle(data, (x1, y1), (x2, y2), (0,0,255),thickness=2)
        elif i in green_area:
            #rect = Rectangle((x1, y1), width, height, fill=False, color='green', lw=2)
            rect = cv2.rectangle(data, (x1, y1), (x2, y2), (0,255,0),thickness=2)
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
	"book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

################ CLIENT
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('localhost', 8485))
connection = client_socket.makefile('wb')

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

os.chdir(r'C:\Users\so16s\AppData\Roaming\JetBrains\PyCharmCE2020.3\scratches\YOLO\darknet')
model = load_model('model.h5',compile=False)
input_w, input_h = 416, 416
video_path = r'C:\Users\so16s\AppData\Roaming\JetBrains\PyCharmCE2020.3\scratches\YOLO\darknet\people_commerce.mp4'
new_frame_path = os.path.join(os.getcwd(), 'new')
load_frames_of_video(video_path, new_frame_path)

for id, im_filename in enumerate(os.listdir(new_frame_path)):
  #if id % 15 == 0 and id >= 100:
    im_filename = os.path.join(new_frame_path, im_filename)
    image, image_w, image_h = load_image_pixels(im_filename, (input_w, input_h))
    image = image.reshape(image.shape[1:])
    #image = cv2.resize(image, (200,200))
    result, frame = cv2.imencode('.jpg', image, encode_param)
    data = pickle.dumps(frame, 0)
    size = len(data)
    time.sleep(1)
    print("{}: {}".format(id, size))
    client_socket.sendall(struct.pack(">L", size) + data)



"""num_imgs = len(os.listdir(new_frame_path))
im_id = 1
plt.figure()
for im_id in range(num_imgs):
    f = os.path.join(new_frame_path, 'frame_{0}.jpg'.format(im_id))
    im = plt.imread(f)
    plt.imshow(im)
    plt.title('Frame %d' % im_id)
    plt.show()
    #time.sleep(0.15)
    clear_output(wait=True)"""
