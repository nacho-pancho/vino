#!/bin/bash
for i in 1 2
do
  mkdir -p results/calib3d/frames/circles_ruler_gopro${i}
  #code/grab_video_frames.py -i data/calibracion/circles_ruler_gopro${i}.mp4 -o results/calib3d/frames/circles_ruler_gopro${i} -r 1 -s 360 -f 385
  #code/grab_video_frames.py -i data/calibracion/circles_ruler_gopro${i}.mp4 -o results/calib3d/frames/circles_ruler_gopro${i} -r 1 -s 800 -f 860
  
  mkdir -p results/calib3d/frames/checkers_ruler_gopro${i}
  code/grab_video_frames.py -i data/calibracion/checkers_ruler_gopro${i}.mp4 -o results/calib3d/checkers_ruler_gopro${i} -r 1 -s 864 -f 1032
  code/grab_video_frames.py -i data/calibracion/checkers_ruler_gopro${i}.mp4 -o results/calib3d/checkers_ruler_gopro${i} -r 1 -s 1152 -f 1344
done
