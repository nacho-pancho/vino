#!/usr/bin/env python3

import ffmpeg
import sys
import os

stream = ffmpeg.input(sys.argv[1]).trim(start_frame=0,end_frame=100).output(sys.argv[2]).run()
