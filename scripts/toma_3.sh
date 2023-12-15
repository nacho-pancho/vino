#!/bin/bash
mkdir results/toma3
for i in 1 2
do
	code/gather_calibration_data.py -i data/2023-12-11-tercera_salida/gopro${i}/gopro${i}_white_3_y_toma_3.mp4 -s 0 -f 120 -o results/toma3/toma3_gopro${i}
	code/process_calibration_data.py -i results/toma3/toma3_gopro${i}_white_frame.png -o results/toma3/toma3_gopro${i} -c 100,1080,200,1500
        code/apply_calibration.py -i data/2023-12-11-tercera_salida/gopro${i}/gopro${i}_white_3_y_toma_3.mp4 -o results/toma3/toma3_gopro${i} -w results/toma3/toma3_gopro${i}_curve.png -b results/toma3/toma3_gopro${i}_white_balance.txt -s 3600 -f 4000 -e 0.3

done
