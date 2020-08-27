# Trabajo-de-Grado---Juan-Mendez
IDENTIFICACIÓN DE INSTRUMENTOS MUSICALES A PARTIR DEL ALGORITMO MFCC (MEL FREQUENCY CEPSTRAL COEFFICIENT)

En el presente repositorio encontrará los diferentes archivos digitales que evidencian los resultados obtenidos
durante el desarrollo del trabajo de grado.

AUDIOS:
Los archivos Clarinete_notas.csv, Flauta_notas.csv, Trombon_notas.csv, Timbal_notas.csv hacen referencia a los  
audios del conjunto de entrenamiento y validación.
El archivo Mixto.csv hace referencia a la lista de audios del conjunto de clasificación.
NOTA: Los audios se encuentran en el link:
sin embargo, para ejecutar los archivos debe descargar los audios y ubicarlo en la misma carpeta en la que se
encuntran los archivos de este repositorio.

Los archivos Caracteristicas.csv y Evaluacion.csv son las caracteristicas extraidas de los conjuntos de 
entrenamiento y d clasificación respectivamente. Estos archivos son necesarios para la ejcución de algunos archivos
como MFCC.py


DESCRIPCIÓN ARCHIVOS:

Funcines.py: Este archivo recoge todas las funciones necesarias para el desarrollo del entrenamiento y clasificación.
Entre estas se encuentran:
    lecturaaudio(audio): Lee los audios en formato wave.
    adicionruido(audio): Lee los audios en formato wave y adiciona un ruido con cierto SNR.
    FactorMel(audio, muestreo): Tiene como entrada la señal del audio previamente leido y la frecuncia de muestreo.
        En su salida se encuentran los primero 13 coeficientes MFCC por cada cuadro, es decir una matriz de 260x13
    Caracteristicas(coeficientes): Recibe la matriz de coeficientes y extre las medidas estidsticas. A su salida se
        obtiene el vector de caracteristicas de 65x1.
    GraficaROC(Clasificador,x,y): Función que grafica la Curva ROC, como entradas se tiene el modelo de clasificación
        y los vectores a clasificar con x como los datos a clasificar y y el vector de etiquetas.
    MejorNVecinos(k_range,X_entranamiento, y_entranamiento,X_evaluacion, y_evaluacion): Evalua parámetro K entre 0 y 20.

Entrenamiento.py: Código con el que se realiza el entramiento, la validación del entrenamiento y la clasificación.
En este algortimo se obtiene el modelo del árbol de decisiones, las matrices de confusión y las graficas ROC.



AnalisisRuido.py: Código con el que se evaluó la exactitud de los modelos con diferente niveles de SNR.
Se ajusta con la variable SNR de la función adicionruido(audio) del archivo Funciones.py

Algoritmo.py: Código del sistema de clasificación final, en donde se transmite la información al usuario y se 
realizan las pruebas de tiempo.
