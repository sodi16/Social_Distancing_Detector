
# load yolov3 model and perform object detection
# based on https://github.com/experiencor/keras-yolo3
import numpy as np
from tensorflow.keras.models import load_model
import cv2
import shutil
from PIL import Image, ImageDraw
import io
import os
import socket
import pylab as pl
import matplotlib
import time


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


def wait_for_acknowledge(client, response):
    amount_received = 0
    amount_expected = len(response)

    msg = str()
    while amount_received < amount_expected:
        try:
            data = client.recv(1500)
            amount_received += len(data)
            msg += data.decode("utf-8")
        except socket.error as socketerror:
            print("Error: ", socketerror)
        # print(msg)
    return msg


########################################################################### CLIENT
# initiate connection
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
os.chdir(r'C:\Users\so16s\AppData\Roaming\JetBrains\PyCharmCE2020.3\scratches\YOLO\darknet')
model = load_model('model.h5',compile=False)

#video_path = r'C:\Users\so16s\AppData\Roaming\JetBrains\PyCharmCE2020.3\scratches\YOLO\darknet\people.mp4'
video_path = r'C:\Users\so16s\AppData\Roaming\JetBrains\PyCharmCE2020.3\scratches\YOLO\darknet\walking.mp4'

new_frame_path = os.path.join(os.getcwd(), 'new')
load_frames_of_video(video_path, new_frame_path)
fileList = [file for file in os.listdir(new_frame_path)]

server_addr = (socket.gethostname(), 2022)
client.connect(server_addr)
print(f"Connected to server!")
client.settimeout(15)  # limit each communication time to 5s

# listening to server command
##############################################
print("Server sending command: \"Start sending image.\"")

client.sendall(bytes("Start sending image.", "utf-8"))

# wait for reply from client
ack_from_client = wait_for_acknowledge(client, "ACK")
if ack_from_client != "ACK":
    raise ValueError('Client does not acknowledge command.')

# Send message to client to notify about sending image
imgCount = len(fileList)
#print("Client sends the number of images to be transfered server.")
client.sendall(bytes(str(imgCount), "utf-8"))

# wait for reply from client
ack_from_client = wait_for_acknowledge(client, "ACK")
if ack_from_client != "ACK":
    raise ValueError('Client does not acknowledge img count.')

print("Client will now send the images.", end='')
res = None
res = matplotlib.image.AxesImage(res)
last_time = time.time()

for id, file in enumerate(fileList):
    if id % 10 == 0:
        img = open(os.path.join(os.getcwd() + r'\new' ,file), 'rb')
        b_img = img.read()
        imgsize = len(b_img)
        client.sendall(bytes(str(imgsize), "utf-8"))
        print(f"\t sending image {file} size of {imgsize}B.")

        #print("Client is now waiting for acknowledge from server.")
        ack_from_client = wait_for_acknowledge(client, "ACK")
        if ack_from_client != "ACK":
            raise ValueError('Client does not acknowledge img size.')

        client.sendall(b_img)
        img.close()
        print(f"Image {file} sent!")


        imgsize = int(wait_for_acknowledge(client, str(3)))
        client.sendall(bytes("ACK", "utf-8"))
        buff = client.recv(imgsize)
        image_buff = Image.open(io.BytesIO(buff))
        t = time.time() - last_time
        print('%.3f' %t ,'seconds to display')
        last_time = time.time()
        image_buff.show()


print("All images received.")
print("Closing connection.")
client.close()

