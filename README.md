# vino: estimación de producción vitivinícola

## Resumen

Este proyecto busca estimar la producción de un viñedo a partir de capturas de imágenes y/o videos de la vid en distintos momentos del año.
Este repositorio contiene código relacionado con el proyecto.

## Estructura general

* `code`: código fuente, principalmente Python
* `data`: aquí van los datos a procesar; no se incluyen en el repo sino que se bajan con scripts
* `scripts`:  scripts de automatización de diversos procesos, incluyendo ejecución batch de programas
* `doc`: documentación técnica relacionada con este repositorio (algoritmos, etc.)

## Demos

* Calibración de luz: scripts/toma_3.sh realiza los cálculos de calibración y los aplica a la toma 3 de la salida 3 (2023-12-11-tercera_salida), a la cámara GoPro 2, que es la que quedó mejor situada

