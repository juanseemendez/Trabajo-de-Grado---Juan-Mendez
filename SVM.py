import numpy as np
import csv

from Funciones import adicionruido
from Funciones import lecturaaudio
from Funciones import FactorMel
from Funciones import caracteristicas
from sklearn.svm import SVC 
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
from sklearn.metrics import confusion_matrix, classification_report


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
        senal_ruido, muestreo=adicionruido(" ".join(fila)) #lectura audio entrada, adicion ruido
        mfccout= FactorMel(senal_ruido, muestreo) #Extracción Factores Mel del audio
        estadisticas_Cla=caracteristicas(mfccout) #Extracción caracteristicas del audio
        Ins_Cla+=[0]      #Asignación clase 0 (Clarinete)
        Clarinete_Caracteristicas+=[estadisticas_Cla]
        
#---------------- Flauta         
with open('Flauta_notas.csv',newline='') as muestra: #Archivo con nombre de los audios
    lector_entrada = csv.reader(muestra,dialect='excel')
    for fila in lector_entrada:
        senal_ruido, muestreo=adicionruido(" ".join(fila)) #lectura audio entrada, adicion ruido
        mfccout= FactorMel(senal_ruido, muestreo) #Extracción Factores Mel del audio
        estadisticas_Fla=caracteristicas(mfccout) #Extracción caracteristicas del audio
        Ins_Fla+=[1]      #Asignación clase 1 (Flauta)
        Flauta_Caracteristicas+=[estadisticas_Fla]

#---------------- Trombon  
with open('Trombon_notas.csv',newline='') as muestra: #Archivo con nombre de los audios
    lector_entrada = csv.reader(muestra,dialect='excel')
    for fila in lector_entrada:      
        senal_ruido, muestreo=adicionruido(" ".join(fila)) #lectura audio entrada, adicion ruido
        mfccout= FactorMel(senal_ruido, muestreo) #Extracción Factores Mel del audio
        estadisticas_Trom=caracteristicas(mfccout) #Extracción caracteristicas del audio
        Ins_Trom+=[2]     #Asignación clase 2 (Trombon)
        Trombon_Caracteristicas+=[estadisticas_Trom]

#---------------- Timbal         
with open('Timbal_notas.csv',newline='') as muestra: #Archivo con nombre de los audios
    lector_entrada = csv.reader(muestra,dialect='excel')
    for fila in lector_entrada:
        senal_ruido, muestreo=adicionruido(" ".join(fila)) #lectura audio entrada, adicion ruido
        mfccout= FactorMel(senal_ruido, muestreo) #Extracción Factores Mel del audio 
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

# ------------------------- CLASIFICADOR
        
x = Caracteristicas  #Conjunto de caracteristicas para entrenamiento (valores estadisticos)
y = Instrumento #Etiquetas (Instrumentos: 0:Clarinete, 1:Flauta, 2:Trombon) para entrenamiento

# Separación del conjunto de datos en un set de entrenamiento y un set de prueba
X_entranamiento, X_evaluacion, y_entranamiento, y_evaluacion = train_test_split(x, y,test_size=0.3,random_state=42)

# Conjunto de datos para la evaluacion del modelo de clasificación
X_ev = np.array(Mix_Caracteristicas) #Características instrumentos mixtos (valores estadisticos)
y_ev = Ins_Mix #Etiquetas (Instrumentos: 0:Clarinete, 1:Flauta, 2:Trombon)

# ------ SVM Linear
SVML = SVC(kernel='linear', C=1E10,gamma='auto') #Creación clasificador SVM
SVML = SVML.fit(X_entranamiento, y_entranamiento) #Entrenamiento del modelo

vectoresL=SVML.support_vectors_  #Vectores de soporte
indicesL=SVML.support_  #Indice de los vectores
nvectorL=SVML.n_support_ #Número de vectores por cada categoria

y_preL= SVML.predict(X_evaluacion) #Predicción instrumentos notas simples
Exactitud_SVML = metrics.accuracy_score(y_evaluacion, y_preL)*100 #Precisión del modelo en entrenamiento 

print('Exactitud de Clasificador SVM Lineal en set de entrenamiento: {:.2f}'
     .format(SVML.score(X_entranamiento, y_entranamiento)))
print('Exactitud de Clasificador SVM Lineal en set de validacion: {:.2f}'
     .format(SVML.score(X_evaluacion, y_evaluacion)))

print(confusion_matrix(y_evaluacion, y_preL))
print(classification_report(y_evaluacion, y_preL))

#-----Clasificación
y_preEL= SVML.predict(X_ev) #Predicción para los instrumentos mixtos
Exactitud_SVMEL = metrics.accuracy_score(y_ev, y_preEL)*100 #Precisión del modelo en  evaluación

print('Exactitud de Clasificador SVM Lineal en evaluacion: {:.2f}'
     .format(SVML.score(X_ev, y_ev)))

print(confusion_matrix(y_ev, y_preEL))
print(classification_report(y_ev, y_preEL))

# ------ SVM Gaussiano
SVMG = SVC(kernel='rbf', C=1E10,gamma='auto') #Creación clasificador SVM
SVMG = SVMG.fit(X_entranamiento, y_entranamiento) #Entrenamiento del modelo

vectoresG=SVMG.support_vectors_  #Vectores de soporte
indicesG=SVMG.support_  #Indice de los vectores
nvectorG=SVMG.n_support_ #Número de vectores por cada categoria

y_preG= SVMG.predict(X_evaluacion) #Predicción instrumentos notas simples
Exactitud_SVMG = metrics.accuracy_score(y_evaluacion, y_preG)*100 #Precisión del modelo en entrenamiento 

print('Exactitud de Clasificador SVM  Gaussiano en set de entrenamiento: {:.2f}'
     .format(SVMG.score(X_entranamiento, y_entranamiento)))
print('Exactitud de Clasificador SVM Gaussiano en set de validacion: {:.2f}'
     .format(SVMG.score(X_evaluacion, y_evaluacion)))

print(confusion_matrix(y_evaluacion, y_preG))
print(classification_report(y_evaluacion, y_preG))

#-----Clasificación
y_preEG= SVMG.predict(X_ev) #Predicción para los instrumentos mixtos
Exactitud_SVMEG = metrics.accuracy_score(y_ev, y_preEG)*100 #Precisión del modelo en  evaluación

print('Exactitud de Clasificador SVM Gaussiano en evaluacion: {:.2f}'
     .format(SVMG.score(X_ev, y_ev)))

print(confusion_matrix(y_ev, y_preEG))
print(classification_report(y_ev, y_preEG))

# ------ SVM Polinomial
SVMP = SVC(kernel='poly',C=1E10,degree=2,coef0=1.0,gamma='auto') #Creación clasificador SVM
SVMP = SVMP.fit(X_entranamiento, y_entranamiento) #Entrenamiento del modelo

vectoresP=SVMP.support_vectors_  #Vectores de soporte
indicesP=SVMP.support_  #Indice de los vectores
nvectorP=SVMP.n_support_ #Número de vectores por cada categoria

y_preP= SVMP.predict(X_evaluacion) #Predicción instrumentos notas simples
Exactitud_SVMP = metrics.accuracy_score(y_evaluacion, y_preP)*100 #Precisión del modelo en entrenamiento 

print('Exactitud de Clasificador SVM Polinomial en set de entrenamiento: {:.2f}'
     .format(SVMP.score(X_entranamiento, y_entranamiento)))
print('Exactitud de Clasificador SVM Polinomial en set de VALIDACION: {:.2f}'
     .format(SVMP.score(X_evaluacion, y_evaluacion)))

print(confusion_matrix(y_evaluacion, y_preP))
print(classification_report(y_evaluacion, y_preP))

#-----Clasificación
y_preEP= SVMP.predict(X_ev) #Predicción para los instrumentos mixtos
Exactitud_SVMEP = metrics.accuracy_score(y_ev, y_preEP)*100 #Precisión del modelo en  evaluación

print('Exactitud de Clasificador SVM Polinomial en evaluacion: {:.2f}'
     .format(SVMP.score(X_ev, y_ev)))

print(confusion_matrix(y_ev, y_preEP))
print(classification_report(y_ev, y_preEP))

