#!/bin/bash

mkdir results/calib3d
for i in 1 2
do
  mkdir -p results/calib3d/gopro${i}
  code/calibrate_camera.py -l data/calibracion/checkers/frames_gopro${i}.list -i data/calibracion/checkers -o results/calib3d/gopro${i}  --nframes 50 --downscale 2 --pattern-size 41.2
done
