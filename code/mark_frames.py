#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
This program takes frames directly from a video file and produces two main outputs:
* a white frame to be used for correcting for non-uniform illumination; this is a grayscale map computed as 0.5G+0.25R+0.25B
* a correction factor to the R G and B factors so that a white balance is defined independently from the camera settings
The user must also provide:
* the starting frame in the video
* the final frame in the video
* a cropping region
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
import tkinter as tk
from tkinter import ttk


class VidUI():
    '''
    Clase principal
    '''
    def __init__(self,cap1,cap2):
        super().__init__()
        self.root = tk.Tk()

        self.style = ttk.Style(self.root)

        self.width = width
        self.height = height
        self.drawing = False
        self.x1 = 0
        self.y1 = 0
        self.x2 = 0
        self.y2 = 0
        self.rot1 = 0
        self.rot2 = 0
        self.initial_calib_frame = -1
        self.initial_white_frame = -1
        self.final_calib_frame = -1
        self.final_white_frame = -1

        self.dtick = round(1000)

        #
        # creamos la interfaz gráfica (ventana)
        #
        self.root.title("Anotador de frames de vid")
        if cap1 is None:
            ncams = 0
            exit(1)
        
        if cap2 is not None:
            self.ncams = 2
        else:
            self.ncams = 1
        
        self.cv2_frame = [None,None]
        self.color_frame = [None,None]
        self.gray_frame = [None,None]
        self.rot = np.zeros(2)
        for C in range(ncams):
            self.fps[C] = self.cap[C].get(cv2.CAP_PROP_FPS)
        self.frame_idx = 0
        self.grab_frame()

        height,width,nchannels = self.color_frame[C].shape
        window_width = 2*width
        window_height = height
        self.root.geometry(f"{window_width}x{window_height}")

        fmain = ttk.Frame(self.root)
        fopt = ttk.Frame(fmain,padding=10)
        self.rot = np.zeros(2)

        self.fwd_button = ttk.Button(fopt,text=">>",padding=3,command=self.next_frame)
        self.fwd_button.grid(row=0,column=0)
        self.fwd_button = ttk.Button(fopt,text="<<",padding=3,command=self.prev_frame)
        self.fwd_button.grid(row=0,column=1)
        self.ini_wf_check = ttk.Button(fopt, text='initial WF',onvalue=1, offvalue=0, command=self.set_initial_white_frame)
        self.ini_wf_check.grid(row=0,column=2)
        self.fin_wf_check = ttk.Button(fopt, text='final WF',onvalue=1, offvalue=0, command=self.set_final_white_frame)
        self.fin_wf_check.grid(row=0,column=3)
        self.ini_calib_check = ttk.Button(fopt, text='initial calib',onvalue=1, offvalue=0, command=self.set_initial_calib_frame)
        self.ini_calib_check.grid(row=0,column=4)
        self.fin_calib_check = ttk.Button(fopt, text='final calib',onvalue=1, offvalue=0, command=self.set_final_calib_frame)
        self.fin_calib_check.grid(row=0,column=5)
        self.ini_calib_check = ttk.Button(fopt, text='left frame time ref',onvalue=1, offvalue=0, command=self.set_time_mark_1)
        self.ini_calib_check.grid(row=0,column=4)
        self.fin_calib_check = ttk.Button(fopt, text='right frame time ref',onvalue=1, offvalue=0, command=self.set_time_mark_2)
        self.fin_calib_check.grid(row=0,column=5)

        fopt.pack(side="bottom")
        #
        # creamos el lienzo (canvas) en donde dibujar
        #
        self.canvas = tk.Canvas(fmain, width=self.width, height=self.height, bg='black')
        
        self.canvas.bind('<ButtonPress-1>', self.mouse_down )
        self.canvas.bind('<ButtonRelease-1>', self.mouse_up )
        self.canvas.pack()
        fmain.pack()
        self.canvas.after(self.dtick,self.update)
        self.root.mainloop()

    def grab_frame(self):
        for C in range(self.ncams):
            self.cap[C].set(cv2.CAP_PROP_POS_FRAMES, self.frame_idx)
            ret, self.cv2_frame[C] = cap[C].read()
            ret, self.cv2_frame[C] = cap[C].read(self.cv2_frame[C])
            self.color_frame[C] = np.flip(np.array(self.cv2_frame[C]),axis=2)
            height,width,nchannels = self.color_frame[C].shape
            print(f"Camera {C}: input frame dimensions: height={height} width={width} channels={nchannels}")        
            if self.rot[C]:
                self.color_frame[C] = 255*trans.rotate(self.color_frame,-self.rot[C],resize=True) # rotation scales colors to 0-1!!
            self.gray_frame[C] = \
                0.25*self.color_frame[C][:,:,0] + \
                0.5*self.color_frame[C][:,:,1] + \
                0.25*self.color_frame[C][:,:,2]


    def mouse_down(self,event):
        self.drawing = True
        self.x1 = self.root.winfo_pointerx()-self.root.winfo_rootx()
        self.y1 = self.root.winfo_pointery()-self.root.winfo_rootx()
        self.y2 = self.y1
        self.x2 = self.x1


    def mouse_up(self,event):
        self.drawing = False
        self.cropbox = [min(self.y1,self.y2), min(self.x1,self.x2),max(self.y1,self.y2),max(self.x1,self.x2)]


    def set_initial_white_frame(self,value):
        self.initial_white_frame = value


    def set_final_white_frame(self,value):
        self.final_white_frame = value


    def set_initial_calib_frame(self,value):
        self.initial_calib_frame = value


    def set_final_calib_frame(self,value):
        self.final_calib_frame = value


    def set_time_mark_1(self,value):
        self.time_mark_1 = value


    def set_time_mark_2(self,value):
        self.time_mark_2 = value


    def clear_screen(self):
        self.canvas.create_rectangle(0,0,self.width,self.height,fill='white')


    def advance_frame(self):
        self.frame_idx += 1
        self.grab_frame()


    def update(self):
        self.x2 = self.root.winfo_pointerx()-self.root.winfo_rootx()
        self.y2 = self.root.winfo_pointery()-self.root.winfo_rooty()
        if self.drawing:
            self.paint()
        self.canvas.after(self.dtick,self.update)


    def paint(self):
        #
        # coordenadas del mouse originales
        #
        w = self.width
        h = self.height
        for C in range(self.ncams):
            self.canvas.create_image(0,0,anchor=cv2.NW,image=self.color_frame[0])
            self.canvas.create_image(0,0,anchor=cv2.NE,image=self.color_frame[0])


if __name__ == "__main__":
    
    ap = argparse.ArgumentParser()
    #
    # mmetadata
    #
    ap.add_argument("--input1", type=str, required=True,
                    help="input video")
    ap.add_argument("--input2", type=str, default=None,
                    help="input video")
    ap.add_argument('-m',"--method", type=str, default="max",
                    help="Method for computing the white frame. May be average,max,or an integer for the percentile (much slower).")
  
    args = vars(ap.parse_args())
    cap = [None,None]
    fps = [None,None]
    cap[0] = cv2.VideoCapture(args["input1"])

    gui  = VidUI(cap[0],cap[1])
    if cap[0]:
        cap[0].release()
    if cap[1]:
        cap[1].release()

