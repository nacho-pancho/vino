#!/usr/bin/env python3
# importing the necessary libraries
import argparse
import sys
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import io as imgio

if len(sys.argv) < 2:
    print('must provide a file name')
    exit(1)
    
cap = cv2.VideoCapture(sys.argv[1])

if len(sys.argv) > 2:
    n0 = int(sys.argv[2])
else:
    n0 = 0

if len(sys.argv) >3:
    n1 = int(sys.argv[3])
else:
    n1 = 10000000

# Loop until the end of the video
u = None
frame = None
n = 0
t0 = time.time()
while (cap.isOpened()):
    # Capture frame-by-frame
    if frame is None:
        ret, frame = cap.read()
    else:
        ret, frame = cap.read(frame)
    if not ret:
        break
    #frame = cv2.resize(frame, (540, 380), fx = 0, fy = 0, interpolation = cv2.INTER_CUBIC)
    #frame = cv2.rotate(frame,90)

    # Display the resulting frame
    #cv2.imshow('Frame', frame)

    x = np.array(frame)
    if u is None:
        u = np.zeros(x.shape[:2])
    
    if n >= n0:
        u = np.maximum(u,np.mean(x,axis=2))
    n += 1
    # using cv2.Gaussianblur() method to blur the video

    # (5, 5) is the kernel size for blurring.
    #gaussianblur = cv2.GaussianBlur(frame, (5, 5), 0) 
    #cv2.imshow('gblur', gaussianblur)
    #cv2.imshow(f'frame{n:05d}', frame)
    #cv2.imshow('frame', frame)

    # define q as the exit button
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    if not n % 60:
        print('frame',n,'fps',n/(time.time()-t0))
    
    if n > n1:
        break
# release the video capture object
cap.release()
plt.figure()
u = np.round(u).astype(np.uint8)
plt.imshow(u)
plt.show()
imgio.imsave('whiteframe.png',u)

# Closes all the windows currently opened.
cv2.destroyAllWindows()

