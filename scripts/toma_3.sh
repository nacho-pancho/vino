#!/bin/bash
mkdir results/toma3
# la que se ve mejor es la 2, que se ve de abajo
for i in 2 # 1 2 
do
	code/gather_calibration_data.py -i data/2023-12-11-tercera_salida/gopro${i}/gopro${i}_white_3_y_toma_3.mp4 -s 0 -f 120 -o results/toma3/toma3_gopro${i}
	code/process_calibration_data.py -i results/toma3/toma3_gopro${i}_white_frame.png -o results/toma3/toma3_gopro${i} -c 100,1080,200,1500
        code/apply_calibration.py -i data/2023-12-11-tercera_salida/gopro${i}/gopro${i}_white_3_y_toma_3.mp4 -o results/toma3/toma3_gopro${i} -w results/toma3/toma3_gopro${i}_curve.png -b results/toma3/toma3_gopro${i}_white_balance.txt -s 2100 -f 3000 -e 0.75 -R 90 -r 15

done
