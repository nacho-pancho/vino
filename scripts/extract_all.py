#!/usr/bin/env python3

import os
import csv
import sys

if __name__ == "__main__":
    csv_path = sys.argv[1]
    with open(csv_path,'r') as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)
        row1 = next(reader)
        fila1,marca1,tiempo1=int(row1[0]),int(row1[1]),int(row1[2])
        for row2 in reader:
            fila2,marca2,tiempo2=int(row2[0]),int(row2[1]),int(row2[2])
            print(f"code/extract.py -D data/videos -A 2024-03-04-vino_fino -a gopro1 -b gopro2 -s 6 -S -i {tiempo1} -f {tiempo2} -o 2023-04-03-vino_fino-sector_{marca2}-{marca1}")
            fila1,marca1,tiempo1=fila2,marca2,tiempo2
            