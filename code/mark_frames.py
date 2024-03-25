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
from PIL import Image, ImageTk
import json

LINE_SPACING = 40
FONT_SIZE = 20

class VidUI():
    '''
    Clase principal
    '''
    def __init__(self,cap1,cap2,outfname,infname):
        super().__init__()
        self.root = tk.Tk()
        self.style = ttk.Style(self.root)
        self.outfname = outfname
        self.infname = infname
        self.dtick = 100

        #
        # creamos la interfaz gráfica (ventana)
        #
        self.root.title("Anotador de frames de vid")
        if cap1 is None:
            self.ncams = 0
            exit(1)
        
        if cap2 is not None:
            self.ncams = 2
        else:
            self.ncams = 1
        
        self.cap = [None,None]
        self.cv2_frame = [None,None]
        self.color_frame = [None,None]
        self.gray_frame = [None,None]
        self.tk_image = [None,None]
        self.fps = [None,None]
        self.rot = 90*np.ones(2)

        self.cap[0] = cap1
        self.cap[1] = cap2
        for C in range(self.ncams):
            self.fps[C] = self.cap[C].get(cv2.CAP_PROP_FPS)
        self.frame_idx = 0
        self.grab_frame()

        height,width,nchannels = self.color_frame[C].shape
        two_images_width = 2 * width
        two_images_height = height
        self.window_width = 1900
        self.window_height = 900
        scale_x = 1
        if self.window_width < two_images_width:
            scale_x = self.window_width/two_images_width
        
        scale_y = 1
        if self.window_height < two_images_height:
            scale_y = self.window_height / two_images_height
        self.scale = min(scale_x,scale_y)
        self.margin = 200
        if self.window_width > self.scale*two_images_width:
            self.window_width = int(self.scale*two_images_width) + 200
        if self.window_height > self.scale*two_images_height:
            self.window_height = int(self.scale*two_images_height) + 50

        self.root.geometry(f"{self.window_width}x{self.window_height}")

        self.image_width = int(width*self.scale)
        self.image_height = int(height*self.scale)
        self.height = self.window_height
        self.drawing = False
        self.x1 = 0
        self.y1 = 0
        self.x2 = 0
        self.y2 = 0
        self.rot1 = 0
        self.rot2 = 0
        self.reset()
        
        fmain = ttk.Frame(self.root)
        fopt = ttk.Frame(fmain,padding=2)

        b = ttk.Button(fopt,text="<<",padding=2,command=self.prev_frame_10)
        b.grid(row=0,column=0)
        b = ttk.Button(fopt,text="<|",padding=2,command=self.prev_frame)
        b.grid(row=0,column=1)
        b = ttk.Button(fopt,text="|>",padding=2,command=self.next_frame)
        b.grid(row=0,column=2)
        b = ttk.Button(fopt,text=">>",padding=2,command=self.next_frame_10)
        b.grid(row=0,column=3)

        b = ttk.Button(fopt, text='time ref 1', padding=2, command=self.set_time_mark_1)
        b.grid(row=0,column=4)
        
        b = ttk.Button(fopt, text='ini WF', padding=2, command=self.set_initial_white_frame)
        b.grid(row=0,column=5)
        b = ttk.Button(fopt, text='fin WF', padding=2, command=self.set_final_white_frame)
        b.grid(row=0,column=6)

        b = ttk.Button(fopt, text='ini calib', padding=2, command=self.set_initial_calib_frame)
        b.grid(row=0,column=7) 
        b = ttk.Button(fopt, text='fin calib', padding=2, command=self.set_final_calib_frame)
        b.grid(row=0,column=8)

        b = ttk.Button(fopt, text='time ref 2', padding=2, command=self.set_time_mark_2)
        b.grid(row=0,column=9)

        b = ttk.Button(fopt, text='save', padding=2, command=self.save)
        b.grid(row=0,column=10)

        b = ttk.Button(fopt, text='reset', padding=2, command=self.reset)
        b.grid(row=0,column=11)

        fopt.pack(side="bottom")
        #
        # creamos el lienzo (canvas) en donde dibujar
        #
        self.canvas = tk.Canvas(fmain, width=self.window_width, height=self.window_height, bg='#222')
        self.canvas.bind('<ButtonPress-1>', self.mouse_down )
        self.canvas.bind('<ButtonRelease-1>', self.mouse_up )
        self.canvas.bind('<Motion>',self.mouse_moved)
        self.canvas.pack(fill=tk.BOTH,expand=1)
        fmain.pack()
        self.canvas.after(self.dtick,self.update)
        self.root.mainloop()


    def grab_frame(self):
        for C in range(self.ncams):
            self.cap[C].set(cv2.CAP_PROP_POS_FRAMES, self.frame_idx)
            ret, self.cv2_frame[C] = cap[C].read()
            self.color_frame[C] = np.flip(np.array(self.cv2_frame[C]),axis=2)
            height,width,nchannels = self.color_frame[C].shape
            if self.rot[C]:
                self.color_frame[C] = 255*trans.rotate(self.color_frame[C],-self.rot[C],resize=True) # rotation scales colors to 0-1!!
            self.gray_frame[C] = \
                0.25*self.color_frame[C][:,:,0] + \
                0.5*self.color_frame[C][:,:,1] + \
                0.25*self.color_frame[C][:,:,2]


    def save(self):
        print("save to ", self.outfname)
        x = {
            "ini_wf" : self.initial_white_frame,
            "fin_wf" : self.final_white_frame,
            "ini_cf" : self.initial_calib_frame,
            "fin_cf" : self.final_calib_frame,
            "time_mark"  : self.time_mark
        }
        txt = json.dumps(x,indent=4)
        with open(self.outfname,"w") as f:
            f.write(txt)
            print(txt)


    def load(self):
        json.loads()


    def reset(self):
        self.initial_calib_frame = -1
        self.initial_white_frame = -1
        self.final_calib_frame = -1
        self.final_white_frame = -1
        self.time_mark = [-1,-1]


    def mouse_down(self,event):
        self.drawing = True
        self.x1 = self.root.winfo_pointerx()-self.root.winfo_rootx()
        self.y1 = self.root.winfo_pointery()-self.root.winfo_rootx()
        self.y2 = self.y1
        self.x2 = self.x1
        print("mouse down")


    def mouse_up(self,event):
        self.drawing = False
        self.cropbox = [min(self.y1,self.y2), min(self.x1,self.x2),max(self.y1,self.y2),max(self.x1,self.x2)]
        print("mouse up",self.cropbox)


    def mouse_moved(self,event):
        if self.drawing:
            self.x2 = max(min(self.root.winfo_pointerx()-self.root.winfo_rootx(),self.image_width+self.margin//3),0)
            self.y2 = max(self.root.winfo_pointery()-self.root.winfo_rootx(),0)
            self.update()


    def set_initial_white_frame(self):
        self.initial_white_frame = self.frame_idx
        print("set initial white frame to ",self.initial_white_frame)
        self.update()


    def set_final_white_frame(self):
        self.final_white_frame = self.frame_idx
        print("set final white frame to ",self.final_white_frame)
        self.update()


    def set_initial_calib_frame(self):
        self.initial_calib_frame = self.frame_idx
        print("set initial calib frame to ",self.initial_calib_frame)
        self.update()


    def set_final_calib_frame(self):
        self.final_calib_frame = self.frame_idx
        print("set final calib frame to ",self.final_calib_frame)
        self.update()


    def set_time_mark_1(self):
        self.time_mark[0] = self.frame_idx
        self.update()


    def set_time_mark_2(self):
        self.time_mark[1] = self.frame_idx
        self.update()


    def next_frame(self):
        self.frame_idx += 1
        self.grab_frame()
        self.update()


    def next_frame_10(self):
        self.frame_idx += 10
        self.grab_frame()
        self.update()


    def prev_frame(self):
        if self.frame_idx > 0:
            self.frame_idx -= 1
            self.grab_frame()
            self.update()


    def prev_frame_10(self):
        if self.frame_idx > 9:
            self.frame_idx -= 10
            self.grab_frame()
            self.update()
 

    def update(self):
        self.paint()
        #self.canvas.after(self.dtick,self.update)


    def paint(self):
        huge_font = ("Arial",24)
        large_font = ("Arial",20)
        normal_font = ("Arial",18)
        #self.canvas.create_rectangle(0,0,self.canvas.winfo_width,self.canvas.winfo_height)
        self.canvas.delete("all")
        for C in range(self.ncams):
            yoff = 20
            self.npimg = trans.rescale(self.color_frame[C],self.scale,channel_axis=2).astype(np.uint8)            
            W = self.image_width+self.margin//3
            pilimage = Image.fromarray(self.npimg)
            self.tk_image[C] = ImageTk.PhotoImage(pilimage)
            self.canvas.create_image(self.margin//3 + C*W,0,anchor=tk.NW,image=self.tk_image[C])
            self.canvas.create_text(20+C*W,yoff,font=huge_font, text=f"Frame {self.frame_idx}",anchor=tk.NW, fill="#fff")
            yoff += LINE_SPACING
            self.canvas.create_text(20+C*W,yoff,anchor=tk.NW,fill="#ff8",text=f"Cam {C}",font=large_font)
            if self.x1 != self.x2 or self.y1 != self.y2:
                j0 = min(self.x1,self.x2)
                j1 = max(self.x1,self.x2)
                i0 = min(self.y1,self.y2)
                i1 = max(self.y1,self.y2)
                self.canvas.create_rectangle(C*W+j0,i0,C*W+j1,i1,fill=None,outline="#0f0",width=2)
            yoff += LINE_SPACING
            if self.initial_calib_frame == self.frame_idx:
                self.canvas.create_text(20+C*W,yoff,text="CI",font=normal_font, anchor=tk.NW,fill="#8f0")
            if self.final_calib_frame == self.frame_idx:
                self.canvas.create_text(20+C*W,yoff,text="CF",font=normal_font, anchor=tk.NW,fill="#f80")
            yoff += LINE_SPACING
            if self.initial_white_frame == self.frame_idx:
                self.canvas.create_text(20+C*W,yoff,text="WI",font=normal_font, anchor=tk.NW,fill="#8f0")
            if self.final_white_frame == self.frame_idx:
                self.canvas.create_text(20+C*W,yoff,text="WF",font=normal_font, anchor=tk.NW,fill="#f80")
            yoff += LINE_SPACING
            if self.time_mark[C] == self.frame_idx:
                self.canvas.create_text(20+C*W,yoff,font=huge_font, anchor=tk.NW, text="T",fill="#8ff")


if __name__ == "__main__":
    
    ap = argparse.ArgumentParser()
    #
    # mmetadata
    #
    ap.add_argument("--input-one", type=str, required=True,
                    help="input video")
    ap.add_argument("--input-two", type=str, default=None,
                    help="input video")
    ap.add_argument("--outfile", type=str, default="calibration_info.json",
                    help="JSON output file.")
    ap.add_argument("--infile",  type=str, default="calibration_info.json",
                    help="JSON input file.")
    
    args = vars(ap.parse_args())
    cap = [None,None]
    fps = [None,None]
    cap[0] = cv2.VideoCapture(args["input_one"])
    if args["input_two"] is not None:
        cap[1] = cv2.VideoCapture(args["input_two"])
    gui  = VidUI(cap[0],cap[1],args["outfile"],args["infile"])
    if cap[0]:
        cap[0].release()
    if cap[1]:
        cap[1].release()

