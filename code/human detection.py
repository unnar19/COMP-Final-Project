import cv2
import os
import time
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import numpy as np

def glow_effect(no_bg):
    # increase contrast
    gray = cv2.cvtColor(no_bg, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    no_bg = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)

    # add scan lines
    pattern = np.zeros_like(no_bg, dtype=np.uint8)
    pattern[::5, :, :] = no_bg[::5, :, :]/7
    no_bg = cv2.addWeighted(no_bg, 1, pattern, 0.6, 0)
    

    # make subject cyan
    no_bg[:, :, 1] = no_bg[:, :, 1]/1.5
    no_bg[:, :, 2] = 0

    # add glow
    glow = cv2.GaussianBlur(no_bg, (101, 101), 0)
    black = np.all(no_bg == 0, axis=-1)
    no_bg[black] = glow[black]

    return no_bg

webcam = cv2.VideoCapture(0)
(widthweb,heightweb) = (int(webcam.get(3)), int(webcam.get(4)))

os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp;buffer_size;2'
cap = cv2.VideoCapture('rtsp://10.1.19.70/video', cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

(width,height) = (int(cap.get(3)), int(cap.get(4)))

human_weights = cv2.CascadeClassifier('code/haarcascade_fullbody.xml')

background_image = cv2.resize(cv2.imread("code/pics/black.png"), (300, 480)) 
segmentor = SelfiSegmentation(0)

x = 0
y = 0
w = width
h = height
bounds = []
avg = (0,0,width,height)
running_avg = 5

last_time = time.time()
ret, frame = cap.read()
while(True):

    now = time.time()
    frames = int(30*(now-last_time))
    for _ in range(frames + 2):
        ret, frame = cap.read()

    frame = cv2.resize(frame,(640,480))

    ret, frame1 = webcam.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY ) 
    humans = human_weights.detectMultiScale(gray, 1.5, 1)
    
    largest = 0

    for (xi,yi,wi,hi) in humans:
        size = w*h
        if size > largest:
            largest = size
            x,y,w,h = (xi,yi,wi,hi)


    bounds.append((x,y,w,h))
    avgx, avgy, avgw, avgh = 0,0,0,0

    iterator = len(bounds)
    if iterator > running_avg:
        iterator = running_avg

    if iterator != 0:

        for i in range(iterator):
            xi,yi,wi,hi = bounds[-i]
            avgx += xi
            avgy += yi
            avgw += wi
            avgh += hi

        avgx = int(avgx/iterator)
        avgy = int(avgy/iterator)
        avgw = int(avgw/iterator)
        avgh = int(avgh/iterator)
            

        cropped = frame[avgy:avgy + avgh, avgx:avgx + avgw]
        resized = cv2.resize(cropped,(300,480))
        #cv2.rectangle(frame,(avgx,avgy),(avgx+avgw,avgy+avgh),(255,0,0),2)
        
        # cut background
        no_bg = segmentor.removeBG(resized, background_image, cutThreshold=0.6)
        
        # add glow effect
        no_bg = glow_effect(no_bg)

        cv2.imshow("Cropped BG removed", no_bg)
         
    last_time = time.time()
    cv2.imshow('frame',frame1)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
