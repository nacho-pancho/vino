#!/usr/bin/env python3
# importing the necessary libraries
import argparse
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage.io as imgio

cap = cv2.VideoCapture(sys.argv[1])

# Loop until the end of the video
u = None
frame = None
n = 0
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
    x = x[::5,::5] 
    x = np.mean(x,axis=2)
    
    if u is None:
        u = np.zeros(x.shape)
    u += x
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
    
# release the video capture object
cap.release()
u *= (1/n)
plt.figure()
plt.imshow(u,cmap=plt.cm.gray)
plt.show()
u = u.astype(np.uint8)


# Closes all the windows currently opened.
cv2.destroyAllWindows()

imgio.imsave('average_frame.png',u)
