#!/bin/bash
for i in 1 2
do
  code/grab_video_frames.py -i data/calibracion/gopro${i}/circulos_regla_gopro${i}.MP4 -o results/calib3d/irculos_regla_gopro${i} -r 1 -s 360 -f 385
  code/grab_video_frames.py -i data/calibracion/gopro${i}/circulos_regla_gopro${i}.MP4 -o results/calib3d/circulos_regla_gopro${i} -r 1 -s 800 -f 860
done
