#!/usr/bin/env python3
# importing the necessary libraries
import argparse
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.fftpack as fft
import dictionary 
import patch_mapping
import time
from sklearn import cluster
from sklearn import preprocessing

cap = cv2.VideoCapture(sys.argv[1])

# Loop until the end of the video
frame = None
padded_frame = None
n = 0
s = 8
w = 8
m = w*w
p = 64
t0 = time.time()
dcts = None
D = np.zeros((p,m))
while (cap.isOpened()):
    # Capture frame-by-frame
    if frame is None:
        ret, frame = cap.read()
    else:
        ret, frame = cap.read(frame)
    if not ret:
        break
    #frame = cv2.resize(frame, (1024, 768), fx = 0, fy = 0, interpolation = cv2.INTER_CUBIC)
    frame_mat = np.mean(np.array(frame),axis=2)
    M,N = frame_mat.shape
    if padded_frame is None:
        ph = patch_mapping.padded_size(M, w,s)
        pw = patch_mapping.padded_size(N, w, s)
        padded_frame = np.zeros((M,N))
    padded_frame[:M,:N] = frame_mat
    n += 1
    patches = patch_mapping.extract(padded_frame,w,s)
    if dcts is None:
        dcts = np.zeros(patches.shape)
    for j in range(len(patches)):
        dcts[j,:] = fft.dct(fft.dct(patches[j,:].reshape((w,w)),axis=1,norm="ortho"),axis=0,norm="ortho").reshape((1,64))
        dcts[j,0] = 0 # kill DC
    #dcts = preprocessing.normalize(dcts)

    model = cluster.KMeans(n_clusters=p,n_init=p).fit(dcts)
    D = model.cluster_centers_
    L = model.labels_
    class_patches = np.ones(patches.shape)
    for j in range(len(L)):
        class_patches[j,:] *= L[j]
    class_map = patch_mapping.stitch(class_patches,w,s,ph,pw)
    #cv2.imshow('frame', padded_frame/255)
    cv2.imshow('frame', class_map/p)
    print('fps:',n/(time.time()-t0))
    # define q as the exit button
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    
mosaic = dictionary.show_bw(D,2,0)
# release the video capture object
cap.release()
plt.figure()
plt.imshow(mosaic)
plt.show()

# Closes all the windows currently opened.
cv2.destroyAllWindows()

