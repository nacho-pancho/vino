#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

if __name__ == "__main__":
    if len(sys.argv) > 1 :
        fname = sys.argv[1]
    else:
        fname="/Users/nacho/workspace/vino/data/2024-03-18-vino_comun/gopro3+gopro4.output/camera1_frame_14295.jpg"
    img = cv2.imread(fname)
    img = img[:,:,1]
    plt.figure()
    plt.imshow(img)
    qr_detector = cv2.QRCodeDetector()
    qr_info, qr_points, qr_data = qr_detector.detectAndDecode(img)
    if qr_info is None or not len(qr_info):   
        print('No QR detected')
        exit()
    qr_points = np.squeeze(np.round(qr_points).astype(int))
    print('qr info:',qr_info)
    print('qr points:',qr_points)
    plt.figure()
    for p in qr_points:
        j,i = p
        img[i-2:i+2,j-2:j+2] = 1
    plt.imshow(img)
    plt.show()
