#--------------ALGORITMO
import numpy as np
import csv

from Funciones import adicionruido
from Funciones import lecturaaudio
from Funciones import FactorMel
from Funciones import caracteristicas
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
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
X_entranamiento, X_evaluacion, y_entranamiento, y_evaluacion = train_test_split(x, y, test_size=0.3, random_state=42)

# Conjunto de datos para la evaluacion del modelo de clasificación
X_ev = np.array(Mix_Caracteristicas) #Características instrumentos mixtos (valores estadisticos)
y_ev = Ins_Mix #Etiquetas (Instrumentos: 0:Clarinete, 1:Flauta, 2:Trombon)

# -------------------------ÁRBOL DE DECISION -----------------------------
#-----Entrenamiento
AD = DecisionTreeClassifier(criterion='entropy') #Creación clasificador AD
AD = AD.fit(X_entranamiento,y_entranamiento) #Entrenamiento del Arbol de decisiones

y_pred = AD.predict(X_evaluacion) #Predicción instrumentos notas simples
Exactitud_AD = metrics.accuracy_score(y_evaluacion, y_pred)*100 #Precisión del modelo en entrenamiento 

print('Exactitud de Clasificador AD en set de entrenamiento: {:.2f}'
     .format(AD.score(X_entranamiento, y_entranamiento)))
print('Exactitud de Clasificador AD en set de prueba: {:.2f}'
     .format(AD.score(X_evaluacion, y_evaluacion)))

print(confusion_matrix(y_evaluacion, y_pred))
print(classification_report(y_evaluacion, y_pred))

#-----Clasificación
y_predE = AD.predict(X_ev)  # Predicción para los instrumentos mixtos
Exactitud_ADEv = metrics.accuracy_score(y_ev, y_predE)*100 # Precisión del modelo en evaluación

print('Exactitud de Clasificador AD en evaluacion: {:.2f}'
     .format(AD.score(X_ev, y_ev)))

print(confusion_matrix(y_ev, y_predE))
print(classification_report(y_ev, y_predE))


# -------------------------- VECINOS K-NN --------------------------------
# -----Entrenamiento
n_neighbors = 2 #Número de vecinos
 
knn = KNeighborsClassifier(n_neighbors) #Creación clasificador K-NN
Knn = knn.fit(X_entranamiento, y_entranamiento) #Entrenamiento del modelo

pred = knn.predict(X_evaluacion) #Predicción instrumentos notas simples
Exactitud_Knn = metrics.accuracy_score(y_evaluacion, pred)*100 #Precisión del modelo en entrenamiento 

print('Exactitud de Clasificador K-NN en set de entrenamiento: {:.2f}'
     .format(knn.score(X_entranamiento, y_entranamiento)))
print('Exactitud de Clasificador K-NN en set de prueba: {:.2f}'
     .format(knn.score(X_evaluacion, y_evaluacion)))

print(confusion_matrix(y_evaluacion, pred))
print(classification_report(y_evaluacion, pred))

# -----Clasificación
predE = knn.predict(X_ev) #Predicción para los instrumentos mixtos
Exactitud_KnnE = metrics.accuracy_score(y_ev, predE)*100 #Precisión del modelo en evaluación

print('Exactitud de Clasificador K-NN en evaluacion: {:.2f}'
     .format(knn.score(X_ev, y_ev)))

print(confusion_matrix(y_ev, predE))
print(classification_report(y_ev, predE))

# -------------------------- SVM --------------------------------
# -----Entrenamiento
random_state = np.random.RandomState(0)
SVM = SVC(kernel='linear', C=1E10,decision_function_shape='ovr',random_state=random_state) #Creación clasificador SVM
SVM = SVM.fit(X_entranamiento, y_entranamiento) #Entrenamiento del modelo

vectores=SVM.support_vectors_  #Vectores de soporte
indices=SVM.support_  #Indice de los vectores
nvector=SVM.n_support_ #Número de vectores por cada categoria

y_pre= SVM.predict(X_evaluacion) #Predicción instrumentos notas simples
Exactitud_SVM = metrics.accuracy_score(y_evaluacion, y_pre)*100 #Precisión del modelo en entrenamiento 

print('Exactitud de Clasificador SVM en set de entrenamiento: {:.2f}'
     .format(SVM.score(X_entranamiento, y_entranamiento)))
print('Exactitud de Clasificador SVM en set de prueba: {:.2f}'
     .format(SVM.score(X_evaluacion, y_evaluacion)))

print(confusion_matrix(y_evaluacion, y_pre))
print(classification_report(y_evaluacion, y_pre))

#-----Clasificación
y_preE= SVM.predict(X_ev) #Predicción para los instrumentos mixtos
Exactitud_SVME = metrics.accuracy_score(y_ev, y_preE)*100 #Precisión del modelo en  evaluación

print('Exactitud de Clasificador SVM en evaluacion: {:.2f}'
     .format(SVM.score(X_ev, y_ev)))

print(confusion_matrix(y_ev, y_preE))
print(classification_report(y_ev, y_preE))


