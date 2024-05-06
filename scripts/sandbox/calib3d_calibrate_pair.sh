#!/bin/bash
mkdir -p results/calib3d/pair
code/calibrate_pair.py -i data/calibracion/checkers -l data/calibracion/checkers/frames_gopro1.list -r data/calibracion/checkers/frames_gopro2.list -o results/calib3d/pair --downscale 2 --pattern-size 41.2

