# Preprocesamiento y extracción de frames

* Autor: Ignacio Ramírez <nacho@fing.edu.uy>
* Fecha: 6 de mayo de 2024

## Propósito

Los videos capturados con las GoPro deben ser calibrados por tres motivos principales:
1. iluminación no unniforme sobre el objetivo 
1. balance de blancos
1. falta de sincronismo entre una cámara y la otra: los frames por defecto no se corresponden al mismo instante

Además, para hacer una eventual reconstrucción 3D con el par stereo de cámaras, debemos capturar cuadros de calibración
con el patrón de calibración que generamos para tal ocasión.

## Funcionamiento

El sistema se divide en tres etapas:
* anotación (`annotate.py`): esta es una interfaz gráfica muy rústica en donde uno especifica los valores anteriores
* calibración (`calibrate.py`): este programa toma el archivo de anotaciones JSON generado por `annotate.py` y calcula parámetros que luego serán aplicados en la calibración
* extracción (`extract.py`): junto con la info de calibración, las anotaciones, y la especificación de un período inicial y final de captura (frames de inicio y fin), una escala (por defecto se reducen los frames a 1/4 de su resolución original) y una tasa de submuestreo de frames (por defecto se toma 1 de cada 5 frames), este programa genera una secuencia de archivos jpeg de alta calidad cuyo nombre es camaraX_NNNNN.jpg, donde X es la camara (1 o 2) y NNNNN es el numero de frame _luego_ de la calibración de tiempo.

### Nota sobre la calibración de tiempo

Para calibrar los tiempos de los frames, se marca un frame en la cámara de la izquierda y un frame de la cámara de la derecha en el programa de anotación.
La idea es que el tiempo mostrado en el cronómetro sea exactamente igual, pero puede diferir en hasta 2 centésimas de segundo. No es gran problema.

Lo que hay que entender es que la calibración implica comenzar a leer los frames de ambas cámaras en distintos _offsets_: si el tiempo de referencia (por ej., 4.82) de la cámara 1 tiene marcado un frame _posterior_ al que aparece el 4.82 en la cámara 2, quiere decir que la cámara 1 comenzó a grabar _antes_ que la cámara 2. Supongamos que el frame de la cámara 1 en donode aparece 4.82 es el 120 y el frame de la cámara 2 es el 80. Entonces los frames de la cámara 1 comienzan a leerse (y _contarse_) a partir de su frame 40, mientras que los frames de la cámara 2 se comienzan a contar desde el ppio.

En el caso contrario (el mismo número aparece primero en la cámara 2), los frames de la cámara 1 se leen desde el ppio y los de la cámara 2 se leen a partir del offset que resulte de la diferencia entre ambos puntos de sincrnización (sync2 - sync1).

### Nota sobre la rotación

Desafortunadamente no tenemos establecida de antemano la orientación de las cámaras que usamos en cada caso. Lo que terminó sucediendo es que debemos, muy comunmente, rotar uno o ambos videos al mostrarlos/anotarlos/extraerlos. Para esto hay dos parámetros de `annotate.py` , `-r` y `-s`  que toman valores positivos múltilos de 90: 0, 90, 180 y 270. Hay que probar, no queda otra.

## Manual de uso detallado

### Anotación

El programa `annotate.py` genera una ayuda escueta si uno le pasa como parámetro `-h`. La salida actual es:

```
usage: annotate.py [-h] [-D DATADIR] -a CAMERA_A [-b CAMERA_B] [-t TOMA]
                   [-p PARTE] -A ADQDIR [-o JSON_FILE] [-r ROTATION1]
                   [-s ROTATION2]

options:
  -h, --help            show this help message and exit
  -D DATADIR, --datadir DATADIR
                        directorio donde se encuentran todos los datos.
  -a CAMERA_A, --camera-a CAMERA_A
                        primera cámara (siempre tiene que estar)
  -b CAMERA_B, --camera-b CAMERA_B
                        segunda cámara (si es un par)
  -t TOMA, --toma TOMA  número de toma
  -p PARTE, --parte PARTE
                        número de parte (en gral. para calibrar usamos siempre
                        la 1)
  -A ADQDIR, --adqdir ADQDIR
                        nombre de directorio de la instancia de adquisicion,
                        por ej: 2024-01-03-vino_fino SIN terminadores (barras)
  -o JSON_FILE, --json-file JSON_FILE
                        Nombre de archivo de JSON con anotaciones. Si no se
                        especifica se genera en base al resto de los
                        parametros.
  -r ROTATION1, --rotation1 ROTATION1
                        rotation of first input.
  -s ROTATION2, --rotation2 ROTATION2
                        rotation of second input.

```

* `-D DATADIR` especifica la _base_  de todos los datos de todas las adquisiciones. Por ejemplo, en mi mac, eso es /Users/home/nacho/workspace/vino/data. Puede ser especificado relativo al comando. Si `annotate.py ` se ejecuta desde la raiz del GIT `vino`, puede simplemente ponerse `-D data`
- `-A ADQDIR` carpeta base de la instancia de adquisición que quiere procesarse. Por ejemplo, `2024-03-18-vino_comun`. Los parámetros `-D` y `-A`  se concatenan para generar la carpeta base en donde se encuentran a su vez las subcarpetas en donde se encuentran los videos, que llevan como nombre la cámara con las que se tomó. 
* `-a CAMERA_A`, `-b CAMERA_B` especifican los nombres base de las cámaras. Esto _determina_ la carpeta en donde se encuentran los videos a procesasr, y el _prefijo_ de los archivos de video dentro de ellas.
* `-t TOMA`, `-p PARTE` por defecto estos parámetros son ambos 1 y a los efectos de la calibración deberían ser siempre 1 salvo que pase algo raro. Estos dos valores _determinan_ el nombre de los archivos de video junto con los datos anteriores. A modo de ejemplo, si se pasa `-D data -A 2024-03-18-vino_comun -a gopro1 -b gopro2 -t 1 -p 1`, los archivos y carpetas quedan definidos así: `data/2024-03-18-vino_comun/gopro1/gopro1_toma1_parte1.mp4`  para el video de la cámara 1 y `data/2024-03-18-vino_comun/gopro2/gopro2_toma1_parte1.mp4` para el video de la segunda cámara.
* `-o JSON_FILE` determina el nomre del archiuvo en donde se almacenan las anotaciones. Si no se especifica (lo que sugerimos), el archivo de anotaciones queda en `DATADIR/ADQDIR/CAMERA_A+CAMERA_B_tomaTOMA.json`. En el ejemplo anterior seria `data/2024-03-18-vino_comun/gopro1+gopro2_toma1.json`
* `-r ROTATION1` especifica la rotación a aplicar a los frames de la camara a. Por defecto 0
* `-s ROTATION2` rotación d elos frames de la cámara b

El programa muestra...

### Calibración

La calibración calcula cuantro cosas (si todo se especifica):
* el blanco medio de la cámara, para ajustar el balance global de blancos; para eso se usa el promedio de los pixeles _no saturados_ de los frames seleccionados en la anotación entre `ini_white` y `fin_white`.
* la curva media de iluminación. Los focos producen una luz no uniforme sobre las uvas y eso hay que corregirlo. Para eso se expone una plancha blanca frente a cada par de cámaras, más o menos a la distancia que aparecen las uvas, durante cierto tiempo, de modo que cubra todo el frame (idealmente). Los pixeles _no saturados_  de estos frames se promedian y se utilizan para ajustar un polinomio de segundo grado. Hay que tener mucho cuidado de que no queden zonas oscuras/no cubiertas en los frames de calbiración de blancos. Para evitar esto tenemos el `crobpox` que se define dibujando en el programa de anotación.

#### Invocación:

Siguiendo con el ejemplo de que los datos están en una ruta relativa a la raíz de este proyecto (en donde se encuentra este README.md), la calibración se realiza de la siguiente manera:

```
code/calibrate.py -D data -A 2024-03-18-vino_comun -a data/2024-03-04-vino_fino/gopro1+gopro2_toma1.json
```

### Extracción

Finalmente, con la información de calibración, el archivo de anotaciones, un frame inicial y uno final especificados por los parámetros `-i` y `-f`, y un paso de avance (skip) entre frame y frame a adquirir (parámetro `-s`, por defecto de valor 5), se extraen frames y se almacenan  como archivos jpeg bajo una carpeta con el mismo nombre que el archivo de anotaciones pero extension `.output`. Siguiendo con el ejemplo, seria `data/2024-03-18-vino_fino/gopro1+gopro2.output/`

#### Invocación mínima

De nuevo con el ejemplo anterior, la linea de comandos mínima (sin parámetros opcionales), sería algo así:
````
code/extract.py -D data -A 2024-03-04-vino_fino -a data/$1/gopro1+gopro2_toma1.json -i 2365 -f 10000
```

