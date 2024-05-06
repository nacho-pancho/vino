#!/usr/bin/env python3
# importing the necessary libraries
import argparse
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage.io as imgio

if len(sys.argv) < 2:
    print(f'usage: {sys.argv[0]} video.mp4')
    exit(1)

cap = cv2.VideoCapture(sys.argv[1])
capture = cap
print("CV_CAP_PROP_FRAME_WIDTH: '{}'".format(capture.get(cv2.CAP_PROP_FRAME_WIDTH))) 
print("CV_CAP_PROP_FRAME_HEIGHT : '{}'".format(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))) 
print("CAP_PROP_FPS : '{}'".format(capture.get(cv2.CAP_PROP_FPS))) 
print("CAP_PROP_POS_MSEC : '{}'".format(capture.get(cv2.CAP_PROP_POS_MSEC))) 
print("CAP_PROP_FRAME_COUNT  : '{}'".format(capture.get(cv2.CAP_PROP_FRAME_COUNT))) 
print("CAP_PROP_BRIGHTNESS : '{}'".format(capture.get(cv2.CAP_PROP_BRIGHTNESS))) 
print("CAP_PROP_CONTRAST : '{}'".format(capture.get(cv2.CAP_PROP_CONTRAST))) 
print("CAP_PROP_SATURATION : '{}'".format(capture.get(cv2.CAP_PROP_SATURATION))) 
print("CAP_PROP_HUE : '{}'".format(capture.get(cv2.CAP_PROP_HUE))) 
print("CAP_PROP_GAIN  : '{}'".format(capture.get(cv2.CAP_PROP_GAIN))) 
print("CAP_PROP_CONVERT_RGB : '{}'".format(capture.get(cv2.CAP_PROP_CONVERT_RGB))) 
  


W0_r = imgio.imread('max_frame_r_approx.png')
W0_g = imgio.imread('max_frame_g_approx.png')
W0_b = imgio.imread('max_frame_b_approx.png')
W0_r = 255/W0_r
W0_g = 255/W0_g
W0_b = 255/W0_b

h,w = W0_r.shape
W = np.zeros((h,w,3))
W[:,:,0] = W0_r
W[:,:,1] = W0_g
W[:,:,2] = W0_b
h1,w1 = h//2,w//2
W1 = W[::2,::2,:]

out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc(*'MP4V'), 60, (w1,h1))

# Loop until the end of the video
frame0 = None
n = 0
while (cap.isOpened()):
    # Capture frame-by-frame    
    if frame0 is None:
        ret, frame0 = cap.read()
    else:
        ret, frame0 = cap.read(frame0)
    if not ret:
        break
    frame1 = cv2.resize(frame0, (w1, h1), fx = 0, fy = 0, interpolation = cv2.INTER_LINEAR)
    frame2 = (np.minimum(frame1 * W1,255)).astype(np.uint8)
    out.write(frame2)
    x1 = np.flip(np.array(frame1),axis=2) # BGR to RGB
    x2 = np.flip(np.array(frame2),axis=2) # BGR to RGB
    if not n % 600:
        cv2.imshow('Frame', frame2)
        plt.figure()
        plt.imshow(x2)
        plt.show()
        print('frame',n)
        imgio.imsave(f'input_{n:05d}.png', x1)
        imgio.imsave(f'output_{n:05d}.png',x2)

    # define q as the exit button
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    n += 1

# release the video capture object
cap.release()
# release video
out.release()
# Closes all the windows currently opened.
cv2.destroyAllWindows()
