#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
This program takes one or two video inputs and already computed calibration information
and extracts pairs of frames at given intervals, correcting the illumination and white balance
using the calibration information.
"""

# importing the necessary libraries
import argparse
import sys
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import io as imgio
from skimage import transform as trans
import json
import os
import csv
from vutils import *



if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    #
    # mmetadata
    #
    ap.add_argument("-i","--input",type=str,required=True,help="video de entrada.")
    ap.add_argument("-o","--output", type=str, required=True,help="CSV de salida")
    args = ap.parse_args()
    csv_fname    = args.output
    input_fname  = args.input
    csv_file = open(csv_fname,'w')
    csv_writer = csv.writer(csv_file,delimiter=',')
    csv_writer.writerow(('frame','data','x1', 'y1', 'x2','y2','x3','y3','x4','y4'))
    cap = cv2.VideoCapture(input_fname)
    nframes= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("number of frames:",nframes)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("input frame height",h)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("input frame width",w)
    frame = np.empty((h,w,3))

    frame_index = 0
    qr_detector = cv2.QRCodeDetector()
    # Loop until the end of the video
    while cap.isOpened(): # ----- loop over frames
        ret, frame = cap.read(frame)
        if not ret:
            print('End of stream reached.')
            break

        h,w,_ = frame.shape
#        color_frame = cv2.resize(frame,(w//res_fac,h//res_fac))
#        color_frame = np.flip(np.array(color_frame),axis=2)            
#        color_frame = fast_rot(color_frame,rot)
#        hr,wr,_ = color_frame.shape
#        color_frame = color_frame.astype(float)
        gray_frame = np.mean(frame,axis=2)        
        try:
            qr_info, qr_points, qr_data = qr_detector.detectAndDecode(gray_frame)
        except:
            qr_info = None
        if qr_info is not None and len(qr_info):
            qr_info = int(qr_info)
            print(f'frame {frame_index:06d}: QR detected: {qr_info:03d}')
            qr_points = np.squeeze(np.round(qr_points).astype(int))
            csv_row = [frame_index,qr_info]
            csv_row.extend(qr_points.ravel().tolist())
            csv_writer.writerow(csv_row)
        frame_index += 1
        print('.',end='')
        if not frame_index % 80:
            print() 
