#!/bin/bash
for i in 1 2
do
  #code/grab_video_frames.py -i data/calibracion/gopro${i}/circulos_regla_gopro${i}.MP4 -o results/calib3d/circulos_regla_gopro${i} -r 1 -s 360 -f 385
  #code/grab_video_frames.py -i data/calibracion/gopro${i}/circulos_regla_gopro${i}.MP4 -o results/calib3d/circulos_regla_gopro${i} -r 1 -s 800 -f 860
  code/grab_video_frames.py -i data/calibracion/gopro${i}/cuadrados_regla_gopro${i}.MP4 -o results/calib3d/cuadrados_regla_gopro${i} -r 1 -s 864 -f 1032
  code/grab_video_frames.py -i data/calibracion/gopro${i}/cuadrados_regla_gopro${i}.MP4 -o results/calib3d/cuadrados_regla_gopro${i} -r 1 -s 1152 -f 1344
done
