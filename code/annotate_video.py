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

LINE_SPACING = 30
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
        # annotated parameters
        #
        self.annotations = {
            "ini_calib_frame":-1,
            "fin_calib_frame":-1,
            "ini_white_frame":-1,
            "fin_white_frame":-1,
            "ini_data_frame":-1,
            "fin_data_frame":-1,
            "sync_1_frame":-1,
            "sync_2_frame":-1
            }

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
        self.margin = 60
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
        fmain = ttk.Frame(self.root)

        nav_bar = ttk.Frame(fmain,padding=2)
        ri = 0
        ci = 0
        b = ttk.Button(nav_bar,text="<<",name="back10",padding=2,command=self.prev_frame_10)
        b.grid(row=ri,column=ci)
        ci += 1
        b = ttk.Button(nav_bar,text="<|",name="back1",padding=2,command=self.prev_frame)
        b.grid(row=ri,column=ci)
        ci += 1
        b = ttk.Button(nav_bar,text="|>",name="fwd1",padding=2,command=self.next_frame)
        b.grid(row=ri,column=ci)
        ci += 1
        b = ttk.Button(nav_bar,text=">>",name="fwd10",padding=2,command=self.next_frame_10)
        b.grid(row=ri,column=ci)
        ci += 1
        b = ttk.Button(nav_bar, text='save', padding=2, command=self.save)
        b.grid(row=ri,column=ci)
        ci += 1
        b = ttk.Button(nav_bar, text='reset', padding=2, command=self.reset)
        b.grid(row=ri,column=ci)
        ci += 1

        nav_bar.pack(side=tk.BOTTOM)


        side_bar = ttk.Frame(fmain,padding=2)
        ri = 0
        ci = 0

        for k in self.annotations.keys():
            name = k
            text = k[:-6].replace("_"," ") # remove the "frame"
            goto_name = "goto_" + name
            b = ttk.Button(side_bar, name=name,text=text,padding=2)
            b.bind("<Button-1>",self.annotate)
            b.bind("<Button-2>",self.deannotate)
            b.grid(row=ri,column=ci)
            b = ttk.Button(side_bar, name=goto_name, text='go', padding=2)
            b.bind("<Button-1>",self.goto_frame)
            b.grid(row=ri,column=ci+1)
            ri += 1
    
        side_bar.pack(side="left")
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
        self.side_bar = side_bar
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
        txt = json.dumps(self.annotations,indent=4)
        with open(self.outfname,"w") as f:
            f.write(txt)
            print(txt)


    def load(self):
        self.annotations = json.loads()


    def reset(self):
        for k in self.annotations.keys():
            self.annotations[k] = -1
        for k,v in self.side_bar.children:
            if k[:2] == "go":
                v["state"] = "disabled"


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


    def annotate(self,event):
        key = event.widget.winfo_name()
        self.annotations[key] = self.frame_idx
        print(f"set {key}  to {self.frame_idx}")
        #event.widget["state"] = tk.NORMAL
        self.update()


    def deannotate(self,event):
        key = event.widget.winfo_name()
        self.annotations[key] = -1
        print(f"set {key}  to {self.frame_idx}")
        #event.widget["state"] = tk.DISABLED
        self.root.update()
        self.update()


    def goto_frame(self,event):
        key = event.widget.winfo_name()[5:]
        idx = self.annotations[key]
        self.frame_idx = idx
        self.grab_frame()
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
        if self.frame_idx > 0:
            self.frame_idx -= min(self.frame_idx,10)
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
            self.npimg = trans.rescale(self.color_frame[C],self.scale,channel_axis=2,order=0,anti_aliasing=False).astype(np.uint8)            
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
            if self.annotations["ini_calib_frame"] == self.frame_idx:
                self.canvas.create_text(20+C*W,yoff,text="INI CALIB",font=normal_font, anchor=tk.NW,fill="#8f0")
            if self.annotations["fin_calib_frame"] == self.frame_idx:
                self.canvas.create_text(20+C*W,yoff,text="END CALIB",font=normal_font, anchor=tk.NW,fill="#f80")
            yoff += LINE_SPACING
            if self.annotations["ini_white_frame"] == self.frame_idx:
                self.canvas.create_text(20+C*W,yoff,text="INI WHITE",font=normal_font, anchor=tk.NW,fill="#8f0")
            if self.annotations["fin_white_frame"] == self.frame_idx:
                self.canvas.create_text(20+C*W,yoff,text="END WHITE",font=normal_font, anchor=tk.NW,fill="#f80")
            yoff += LINE_SPACING
            if self.annotations["ini_data_frame"] <= self.frame_idx and self.frame_idx <= self.annotations["fin_data_frame"]:
                self.canvas.create_text(20+C*W,yoff,text="DATA",font=normal_font, anchor=tk.NW,fill="#f8f")
            yoff += LINE_SPACING

            if self.annotations[f"sync_{C+1}_frame"] == self.frame_idx:
                self.canvas.create_text(20+C*W,yoff,font=huge_font, anchor=tk.NW, text="SYNC",fill="#8ff")


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

