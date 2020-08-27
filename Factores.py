import numpy as np
import pandas as pd
import csv

from Funciones import adicionruido
from Funciones import lecturaaudio
from Funciones import FactorMel
from Funciones import caracteristicas


Clarinete_Caracteristicas=[]; Flauta_Caracteristicas=[];
Trombon_Caracteristicas=[];   Timbal_Caracteristicas=[];
Mix_Caracteristicas=[]
Instrumento=[]; 
Ins_Cla=[];Ins_Fla=[];Ins_Trom=[];Ins_Tim=[];Ins_Mix=[];
y=0

#---------------- Clarinete           
with open('Clarinete_notas.csv',newline='') as muestra: #Archivo con nombre de los audios
    lector_entrada = csv.reader(muestra,dialect='excel')
    for fila in lector_entrada: 
        senal_entrada, muestreo=adicionruido(" ".join(fila)) #lectura audio entrada, adicion ruido
        mfccout= FactorMel(senal_entrada, muestreo) #Extracción Factores Mel del audio
        estadisticas_Cla=caracteristicas(mfccout) #Extracción caracteristicas del audio
        Ins_Cla+=[0]      #Asignación clase 0 (Clarinete)
        Clarinete_Caracteristicas+=[estadisticas_Cla]
        
#---------------- Flauta         
with open('Flauta_notas.csv',newline='') as muestra: #Archivo con nombre de los audios
    lector_entrada = csv.reader(muestra,dialect='excel')
    for fila in lector_entrada:
        senal_entrada, muestreo=adicionruido(" ".join(fila)) #lectura audio entrada, adicion ruido
        mfccout= FactorMel(senal_entrada, muestreo) #Extracción Factores Mel del audio
        estadisticas_Fla=caracteristicas(mfccout) #Extracción caracteristicas del audio
        Ins_Fla+=[1]      #Asignación clase 1 (Flauta)
        Flauta_Caracteristicas+=[estadisticas_Fla]

#---------------- Trombon  
with open('Trombon_notas.csv',newline='') as muestra: #Archivo con nombre de los audios
    lector_entrada = csv.reader(muestra,dialect='excel')
    for fila in lector_entrada:      
        senal_entrada, muestreo=adicionruido(" ".join(fila)) #lectura audio entrada, adicion ruido
        mfccout= FactorMel(senal_entrada, muestreo) #Extracción Factores Mel del audio
        estadisticas_Trom=caracteristicas(mfccout) #Extracción caracteristicas del audio
        Ins_Trom+=[2]     #Asignación clase 2 (Trombon)
        Trombon_Caracteristicas+=[estadisticas_Trom]

#---------------- Timbal         
with open('Timbal_notas.csv',newline='') as muestra: #Archivo con nombre de los audios
    lector_entrada = csv.reader(muestra,dialect='excel')
    for fila in lector_entrada:
        senal_entrada, muestreo=adicionruido(" ".join(fila)) #lectura audio entrada, adicion ruido
        mfccout= FactorMel(senal_entrada, muestreo) #Extracción Factores Mel del audio 
        estadisticas_Tim=caracteristicas(mfccout) #Extracción caracteristicas del audio
        Ins_Tim+=[3]      #Asignación clase 3 (Timbal)
        Timbal_Caracteristicas+=[estadisticas_Tim]

# Creación conjunto de datos de entrenamiento
Instrumento=np.concatenate((Ins_Cla,Ins_Fla,Ins_Trom,Ins_Tim),axis=None) #Clases o Etiquetas
Caracteristicas=np.concatenate((Clarinete_Caracteristicas,Flauta_Caracteristicas, #Caracteristicas
                                Trombon_Caracteristicas,Timbal_Caracteristicas), axis=0)

#---------------- Evaluación         
with open('Mixto.csv',newline='') as muestra: #Archivo con nombre de los audios
    lector_entrada = csv.reader(muestra,dialect='excel')
    for fila in lector_entrada: 
        senal_entrada, muestreo=lecturaaudio(" ".join(fila)) #lectura audio entrada, adicion ruido
        mfccout= FactorMel(senal_entrada, muestreo)
        estadisticas_Mix=caracteristicas(mfccout) #Extracción caracteristicas del audio
        if (0<=y<30):
            Ins_Mix+=[0]      #Asignación clase 0 (Clarinete)
        elif (30<=y<60):
            Ins_Mix+=[1]      #Asignación clase 1 (Flauta)
        elif (60<=y<90):
            Ins_Mix+=[2]      #Asignación clase 2 (Trombon)
        elif (y>=90):
            Ins_Mix+=[3]      #Asignación clase 3 (Timbal)
        Mix_Caracteristicas+=[estadisticas_Mix]
        y+=1

Factores=np.insert(Caracteristicas, 65, Instrumento, axis=1) #Conjunto de datos
Nombres=[]
for x in range(1,66): #Creación nombre de las columnas para el Dataframe
    Nombres+=['Caracteristica '+str(x)]
Nombres+=['Instrumento']

dfen = pd.DataFrame(data=Factores[0:,0:],columns=Nombres)
dfen.to_csv('Caracteristicas.csv',index=False)

Evaluacion=np.insert(Mix_Caracteristicas, 65, Ins_Mix, axis=1) #Conjunto de datos

dfev = pd.DataFrame(data=Evaluacion[0:,0:],columns=Nombres)
dfev.to_csv('Evaluacion.csv',index=False)









