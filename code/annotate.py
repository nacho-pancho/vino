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
import os
from tkinter import font as tkfont

LINE_SPACING = 30
FONT_SIZE = 20

class VidUI():
    '''
    Clase principal
    '''
    def __init__(self,cap1,cap2,args):
        super().__init__()
        rot1 = args["rotation1"]
        rot2 = args["rotation2"]
        json_fname = args["json_file"]        
        self.root = tk.Tk()
        self.style = ttk.Style(self.root)
        self.json_fname = json_fname
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
        

        self.cap = [cap1,cap2]
        self.rot = [rot1,rot2]

        self.cv2_frame = [None,None]
        self.color_frame = [None,None]
        self.gray_frame = [None,None]
        self.tk_image = [None,None]
        self.fps = [None,None]
        self.nframes = 0
        for C in range(self.ncams):
            self.fps[C] = self.cap[C].get(cv2.CAP_PROP_FPS)
            self.nframes = max(self.nframes,self.cap[C].get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_idx = 0
        self.grab_frame()

        self.frame_height,self.frame_width,nchannels = self.color_frame[C].shape
        two_images_width = 2 * self.frame_width
        two_images_height = self.frame_height
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

        self.scaled_frame_width = int(self.frame_width*self.scale)
        self.scaled_frame_height = int(self.frame_height*self.scale)
        self.height = self.window_height
        print("input frame shape h=",self.frame_height, "w=",self.frame_width)
        print("scaled frame shape h=",self.scaled_frame_height, "w=",self.scaled_frame_width)
        print("scale=",self.scale)
        self.drawing = False
        self.speed = 25
        self.x1 = 0
        self.y1 = 0
        self.x2 = 0
        self.y2 = 0


        #
        # annotated parameters
    
        self.annotations = {
            "camera_a":args["camera_a"],
            "camera_b":args["camera_b"],
            "ini_calib_frame":-1,
            "fin_calib_frame":-1,
            "ini_white_frame":-1,
            "fin_white_frame":-1,
            "ini_data_frame":-1,
            "fin_data_frame":-1,
            "sync_1_frame":-1,
            "sync_2_frame":-1,
            "crop_box": [0,self.frame_height,0,self.frame_width],
            "rot1": self.rot[0],
            "rot2": self.rot[1]
            }

        helv = tkfont.Font(family='Helvetica', size=24, weight='bold')

        self.scaled_cropbox = [self.scale*x for x in self.annotations["crop_box"]]
        fmain = ttk.Frame(self.root)
        nav_bar = ttk.Frame(fmain,padding=2)
        ri = 0
        ci = 0
        basedir,_ = os.path.split(__file__)
        b = ttk.Button(nav_bar,text="<",image=tk.PhotoImage(file=os.path.join(basedir,"icons/back.png")),name="back1",padding=2,command=self.prev_frame)
        #b = tk.Button(nav_bar,text="<",image=tk.PhotoImage(file=os.path.join(basedir,"icons/back.png")),name="back1",command=self.prev_frame,font=helv)
        b.grid(row=ri,column=ci)
        ci += 1
        b = ttk.Button(nav_bar,text=">",image=tk.PhotoImage(file=os.path.join(basedir,"icons/forward.png")),name="fwd1",padding=2,command=self.next_frame)
        b.grid(row=ri,column=ci)
        ci += 1
        b = ttk.Button(nav_bar,text="-",image=tk.PhotoImage(os.path.join(basedir,"icons/down.png")),name="slower",padding=2,command=self.slower)
        b.grid(row=ri,column=ci)
        ci += 1
        self.speed_label = ttk.Label(nav_bar,text=f"{self.speed}",name="speed",padding=2)
        self.speed_label.grid(row=ri,column=ci)
        ci += 1
        b = ttk.Button(nav_bar,text="+",image=tk.PhotoImage(os.path.join(basedir,"icons/up.png")),name="faster",padding=2,command=self.faster)
        b.grid(row=ri,column=ci)
        ci += 1
        b = ttk.Button(nav_bar, text='save', padding=2, command=self.save)
        b.grid(row=ri,column=ci)
        ci += 1
        b = ttk.Button(nav_bar, text='reset', padding=2, command=self.reset)
        b.grid(row=ri,column=ci)
        ci += 1

        nav_bar.pack(side=tk.BOTTOM)
        self.slider = ttk.Scale(fmain,from_=0,to=self.nframes,variable=0,orient='horizontal',command=self.slide_to_frame)
        self.slider.pack(side=tk.TOP,fill='x')

        side_bar = ttk.Frame(fmain,padding=2)
        ri = 0
        ci = 0

        for k in self.annotations.keys():
            if k[-5:] != 'frame':
                continue
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
        print("save to ", self.json_fname)
        self.annotations["crop_box"] = [int(x/self.scale) for x in self.scaled_cropbox]
        txt = json.dumps(self.annotations,indent=4)
        with open(self.json_fname,"w") as f:
            f.write(txt)
            print(txt)


    def load(self):
        with open(self.json_fname,"r") as f:
            self.annotations = json.loads(f.read())
        self.scaled_cropbox = [int(self.scale*x) for x in self.annotations["crop_box"]]
        self.rot[0] = self.annotations["rot1"]
        self.rot[1] = self.annotations["rot2"]


    def reset(self):
        for k in self.annotations.keys():
            if k[:5] == "frame":
                self.annotations[k] = -1
        self.scaled_cropbox =[0,self.scaled_frame_height,0,self.scaled_frame_width]
        for k,v in self.side_bar.children:
            if k[:2] == "go":
                v["state"] = "disabled"


    def mouse_down(self,event):
        self.drawing = True
        self.x1 = self.canvas.winfo_pointerx()-self.canvas.winfo_rootx()
        self.y1 = self.canvas.winfo_pointery()-self.canvas.winfo_rooty()
        self.y2 = self.y1
        self.x2 = self.x1
        print("mouse down")


    def mouse_up(self,event):
        self.drawing = False
        self.scaled_cropbox = [max(0,min(self.y1,self.y2)), 
                               max(0,min(self.x1,self.x2)),
                               min(self.scaled_frame_height,max(self.y1,self.y2)),
                               min(self.scaled_frame_width,max(self.x1,self.x2))]
        print("mouse up. scaled_cropbox",self.scaled_cropbox)

    def mouse_moved(self,event):
        if self.drawing:
            self.x2 = max(min(self.canvas.winfo_pointerx()-self.canvas.winfo_rootx(),self.scaled_frame_width+self.margin//3),0)
            self.y2 = max(self.canvas.winfo_pointery()-self.canvas.winfo_rooty(),0)
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
        idx = min(self.nframes,max(0,idx))
        self.frame_idx = idx
        self.grab_frame()
        self.slider.set(self.frame_idx)
        self.update()


    def slide_to_frame(self,event):
        print("slide to frame", int(self.slider.get()))
        self.frame_idx = int(self.slider.get())
        self.grab_frame()
        self.update()


    def next_frame(self):
        delta = min(self.nframes-self.frame_idx-1,self.speed)
        self.frame_idx += self.speed
        self.slider.set(self.frame_idx)
        self.grab_frame()
        self.update()


    def prev_frame(self):
        delta = min(self.speed,self.frame_idx)
        self.frame_idx -= delta
        self.slider.set(self.frame_idx)
        self.grab_frame()
        self.update()

    def faster(self):
        self.speed *= 5
        self.speed_label['text'] = f"{self.speed:4d}"
        self.update()

    def slower(self):
        if self.speed >= 5:
            self.speed //= 5
        self.speed_label['text'] = f"{self.speed:4d}"
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
            W = self.scaled_frame_width+self.margin//3
            pilimage = Image.fromarray(self.npimg)
            self.tk_image[C] = ImageTk.PhotoImage(pilimage)
            self.canvas.create_image(self.margin//3 + C*W,0,anchor=tk.NW,image=self.tk_image[C])
            self.canvas.create_text(self.margin//3 + 20+C*W,yoff,font=huge_font, text=f"Frame {self.frame_idx}",anchor=tk.NW, fill="#fff")
            yoff += LINE_SPACING
            self.canvas.create_text(20+C*W,yoff,anchor=tk.NW,fill="#ff8",text=f"Cam {C}",font=large_font)
            if self.x1 != self.x2 or self.y1 != self.y2:
                j0 = min(self.x1,self.x2)
                j1 = max(self.x1,self.x2)
                i0 = min(self.y1,self.y2)
                i1 = max(self.y1,self.y2)
                self.canvas.create_rectangle(self.margin//3 + C*W+j0,i0,C*W+j1,i1,fill=None,outline="#4e4",width=1)
            else:
                i0,i1,j0,j1 = self.scaled_cropbox
                self.canvas.create_rectangle(self.margin//3 + C*W+j0,i0,C*W+j1,i1,fill=None,outline="#0f0",width=2)
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
    ap.add_argument("-D","--datadir",type=str,default=".",help="directorio donde se encuentran todos los datos.")
    ap.add_argument("-a","--cam-a", type=str, required=True,
                    help="primera cámara (siempre tiene que estar)")
    ap.add_argument("-b","--cam-b", type=str, default=None,
                    help="segunda cámara (si es un par)")
    ap.add_argument("-t","--toma", type=int, default=1,
                    help="número de toma")
    ap.add_argument("-p","--parte", type=int, default=1,
                    help="número de parte (en gral. para calibrar usamos siempre la 1)")
    ap.add_argument("-A","--adqdir", type=str, required=True,
                    help="nombre de directorio de la instancia de adquisicion, por ej: 2024-01-03-vino_fino SIN terminadores (barras)")
    ap.add_argument("-o","--json-file", type=str, default=None,
                    help="Nombre de archivo de JSON con anotaciones. Si no se especifica se genera en base al resto de los parametros.")
    ap.add_argument("-r","--rotation1", type=int, default=0,
                    help="rotation of first input.")
    ap.add_argument("-s","--rotation2", type=int, default=0,
                    help="rotation of second input.")
    
    args = vars(ap.parse_args())
    cap = [None,None]
    fps = [None,None]
    adq_path = os.path.join(args["datadir"],args["adqdir"])
    print(f"Ruta absoluta de adquisicion: {adq_path}")
    
    camera_a = args["camera_a"]
    toma = args["toma"]
    parte = args["parte"]

    camera_a_path = os.path.join(adq_path,camera_a)
    print(f"Ruta a cámara {camera_a}: {camera_a_path}")
    toma_a_path = os.path.join(camera_a_path,f"{camera_a}_toma{toma}_parte{parte}.mp4")
    print(f"Ruta a toma de cámara {camera_a}: {toma_a_path}")
    cap[0] = cv2.VideoCapture(toma_a_path)
    if cap[0] is None or not cap[0].isOpened():
        print(f"Error al abrir archivo de video {toma_a_path}")
        exit(1)

    camera_b = args["camera_b"]
    if camera_b is not None:
        camera_b_path = os.path.join(adq_path,camera_b)
        print(f"Ruta a cámara {camera_b}: {camera_b_path}")
        toma_b_path = os.path.join(camera_b_path,f"{camera_b}_toma{toma}_parte{parte}.mp4")
        print(f"Ruta a toma de cámara {camera_b}: {toma_b_path}")
        cap[1] = cv2.VideoCapture(toma_b_path)
        if cap[1] is None or not cap[1].isOpened():
            print(f"Error al abrir archivo de video {toma_b_path}")
            exit(1)

    if args["json_file"] is None:
        #
        # asumimos estructura goproN/goproN_tomaM.mp4 para input y de ahi sacamos raiz
        #
        if camera_b is not None:
            json_path = os.path.join(adq_path,f"{camera_a}+{camera_b}_toma{toma}.json")
        else:
            json_path = os.path.join(adq_path,f"{camera_a}_toma{toma}.json")
    else:
        json_path = args["json_file"]
    args["json_file"] = json_path
    print("json file:",json_path)
    gui  = VidUI(cap[0],cap[1],args)
    
    if cap[0]:
        cap[0].release()
    if cap[1]:
        cap[1].release()

