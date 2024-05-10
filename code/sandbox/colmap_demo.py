#!/usr/bin/env python3
import pycolmap
import os
import sys
import csv
import json
import numpy as np

data_dir  = './data'
videos_dir = os.path.join(data_dir,"videos")
frames_dir  = os.path.join(data_dir,'frames')
instance   = '2024-03-04-vino_fino'
instance_dir = os.path.join(frames_dir,instance)
sector = "sector_127-128"
pano_dir = os.path.join(data_dir,'panoramas')
output_dir = os.path.join(pano_dir,instance,sector)
input_dir = os.path.join(instance_dir,sector)
camera = 'gopro1'
take = '1'

video_instance_dir = os.path.join(videos_dir,instance)
frames_instance_dir = os.path.join(frames_dir,instance)
calib_instance_dir = os.path.join(video_instance_dir,'gopro1+gopro2_toma1.calib')
print("-"*80)
print("PATHS ")
print("-"*80)
print("\tVideos:",video_instance_dir)
print("\tInput frames:",input_dir)
print("\tCalibration data:",calib_instance_dir)
print("\tPanorama dir:",pano_dir)
if not os.path.exists(input_dir):
    print("Input directory",input_dir,"not found.")
    exit(1)

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

os.makedirs(output_dir,exist_ok=True)

mvs_path = os.path.join(pano_dir,"mvs")
database_path = os.path.join(pano_dir,"colmap.db")

pycolmap.extract_features(database_path, input_dir)
pycolmap.match_exhaustive(database_path)
maps = pycolmap.incremental_mapping(database_path, input_dir, output_dir)
maps[0].write(output_dir)
#
# dense reconstruction
#
pycolmap.undistort_images(mvs_path, output_dir, input_dir)
#pycolmap.patch_match_stereo(mvs_path)  # requires compilation with CUDA
#pycolmap.stereo_fusion(os.path.join(mvs_path,"dense.ply"), mvs_path)