#!/usr/bin/env python3
import pycolmap
import os
import sys
import csv
import json
import numpy as np

data_dir  = '/Volumes/nacho/datos/vino'
video_dir  = os.path.join(data_dir,'video')
video_dir = data_dir
pano_dir = os.path.join(data_dir,'panorama')
pano_images_dir = os.path.join(pano_dir,'panorama')
frames_dir = os.path.join(data_dir,'frames')
csv_dir    = os.path.join(data_dir,'marcas')
instance   = '2024-03-04-vino_fino'

camera = 'gopro1'
take = '1'

video_instance_dir = os.path.join(video_dir,instance)
frames_instance_dir = os.path.join(frames_dir,instance)
split_frames_dir = os.path.join(frames_instance_dir,'split')
csv_instance_dir = os.path.join(csv_dir,instance)
calib_instance_dir = os.path.join(video_instance_dir,'gopro1+gopro2_toma1.calib')
print("-"*80)
print("PATHS ")
print("-"*80)
print("\tVideos:",video_instance_dir)
print("\tFrames:",frames_instance_dir)
print("\tSplit frames:",split_frames_dir)
print("\tCSV metadata:",csv_instance_dir)
print("\tCalibration data:",calib_instance_dir)
print("\tPanorama dir:",pano_images_dir)


os.makedirs(pano_dir,exist_ok=True)
os.makedirs(pano_images_dir,exist_ok=True)
os.makedirs(split_frames_dir,exist_ok=True)

calib_path = os.path.join(calib_instance_dir,'calibration.json')
with open(calib_path,'r') as calib_file:
    calib_data = json.load(calib_file)
print("-"*80)
print("CALIBRATION DATA")
print("-"*80)
print(json.dumps(calib_data,indent="  "))
fps = calib_data["camera1"]["fps"]

print("-"*80)
print("MARKS")
print("-"*80)
csv_path = os.path.join(csv_instance_dir,f'{camera}_toma{take}.csv')
with open(csv_path,'r') as csv_file:
    marks = list()
    csv_reader = csv.reader(csv_file,delimiter=',')
    header = next(csv_reader)
    print(header)
    for r in csv_reader:
        fila,numero,tiempo_s = int(r[0]), int(r[1]), int(r[2])
        frame_n = int(np.round(tiempo_s*fps))
        marks.append((fila,numero,tiempo_s,frame_n))
print(marks)

mvs_path = os.path.join(pano_dir,"mvs")
database_path = os.path.join(pano_dir,"colmap.db")

pycolmap.extract_features(database_path, frame_dir)
pycolmap.match_exhaustive(database_path)
maps = pycolmap.incremental_mapping(database_path, frame_dir, output_dir)
maps[0].write(output_dir)
#
# dense reconstruction
#
pycolmap.undistort_images(mvs_path, output_dir, frame_dir)
#pycolmap.patch_match_stereo(mvs_path)  # requires compilation with CUDA
#pycolmap.stereo_fusion(os.path.join(mvs_path,"dense.ply"), mvs_path)