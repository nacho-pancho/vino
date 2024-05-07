#!/bin/bash
#
# primer paso: anotar el video
# se anotan 4 cusas:
# * el frame de inicio (en la camara 1, que aparece a la izquierda) para usar en la calibracion del blanco
# * el frame de inicio para la calibracion 3D. Este tiene que ser el primer cuadro en donde se ve el patron de calibracion 
#   completo en el frame de la izquierda
# * frames de final para balance de blanco y para calibracion, siempre relativos a los frames de la izquierda
# * sincronismo: elegir un frame cualquiera de la izquierda y mirar el tiempo que muestra el cron'ometro; a ese frame
#   se lo marca como sync 1
# * luego hay que avanzar/retroceder hasta que aparezca el mismo numero en el frame de la _derecha_; ahi presionar sync 2
# * es importante marcar un 'cropbox' que garantice que todos los pixeles, en ambas camaras, durante el tiempo de calibracion
#   (entre ini white y fin  white). Para eso arrastrar el mouse en el frame _izquierdo_ de modo que cubra la zona de interes
# * cuando todo se haya hecho, hay que guardar los resultados usando el boton 'save'
#   esto genera un archivo en la carpeta bajo data que fue especificada con el parametro -A pasado en la linea de comandos
#   cuyo nombre es la concatenacion de los parametros -a y -b pasados y con terminacion .json
#   si el -A parametro es, por ejemplo, 2024-03-18-vino_fino, la camara a es gopro1 y la camara b es gopro2
#   y la toma es la no. 1 (por defecto) entonces el archivo JSON generado queda en
#   data/2024-03-18-vino_fino/gopro1+gopro2_toma1.json
#
code/annotate.py -D data -A $1 -a gopro1 -b gopro2 -r 90 -s 90
#
# este script toma las anotaciones y ajuata parámetros de iluminación (white blance, white frame, 3D calibration)
# estos son guardados en una carpeta con mismo nombre que el json generado con annotate pero con terminación '.calib'
# siguiendo el ejemplo anterior, seria data/2024-03-18-vino_fino/gopro1+gopro2.calib
#
code/calibrate.py -D data -A $1 -a data/$1/gopro1+gopro2_toma1.json
#
# finalmente, con la información de calibración, el archivo de anotaciones, un frame inicial y uno final (especificados por -i y -f)
# se extraen frames (por defecto cada 5) como archivos jpeg bajo una carpeta con el mismo nombre que el archivo de anotaciones pero extension
# .output.
# siguiendo con el ejemplo, seria data/2024-03-18-vino_fino/gopro1+gopro2.output/
#
code/extract.py -D data -A $1 -a data/$1/gopro1+gopro2_toma1.json -i 1000 -f 4000