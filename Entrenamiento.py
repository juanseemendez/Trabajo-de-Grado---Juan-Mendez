from sklearn import tree      
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC 
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
from sklearn.metrics import confusion_matrix, classification_report
from graphviz import Source
from Funciones import GraficaROC
from Funciones import MejorNVecinos

import pandas as pd
import numpy as np

#----------- SET DE ENTRENAMIENTO
MFCC= pd.read_csv("Caracteristicas.csv", header=None) # Se leen las caracteristicas
a=MFCC
caracteristicas=MFCC.loc[0,:] # Se determina nombre de los índices de las características
MFCC= pd.read_csv("Caracteristicas.csv", header=None, names=caracteristicas)

# Se divide el conjunto de datos en características y variables objetivo (clases o etiquetas)
x = np.array(MFCC[caracteristicas]) # Características (valores estadísticos)
x = x[1:,:65] 
y = a.loc[1:,65] # Etiquetas o variables objetivo (Instrumentos: 0:Clarinete, 1:Flauta, 2:Trombón, 3:Timbales)
    
# Se divide el conjunto de datos en un set de entrenamiento y un set de validación
X_entrenamiento, X_evaluacion, y_entrenamiento, y_evaluacion = train_test_split(x, y, test_size=0.3, random_state=42)

#----------- SET DE CLASIFICACIÓN
Evaluacion= pd.read_csv("Evaluacion.csv", header=None) # Se leen las características
t=Evaluacion
caracteristicasE=Evaluacion.loc[0,:] # Se determina nombre de los índices de las características
Evaluacion= pd.read_csv("Evaluacion.csv", header=None, names=caracteristicasE)

x_evaluacion = np.array(Evaluacion[caracteristicasE]) # Características (valores estadísticos)
X_ev = x_evaluacion[1:,:65] # Elimina la fila con nombre de los índices  
y_ev = t.loc[1:,65] # Etiquetas o variables objetivo (Instrumentos: 0:Clarinete, 1:Flauta, 2:Trombon, 3:Timbales)


# -------------------------ÁRBOL DE DECISION -----------------------------
#-----Entrenamiento
AD = DecisionTreeClassifier(criterion='entropy') # Creación clasificador AD
AD = AD.fit(X_entrenamiento,y_entrenamiento) # Entrenamiento del Árbol de decisiones

y_pred = AD.predict(X_evaluacion) #Predicción instrumentos notas simples
Exactitud_AD = metrics.accuracy_score(y_evaluacion, y_pred)*100 #Precisión del modelo en entrenamiento 

print('Exactitud de Clasificador AD con set de entrenamiento: {:.2f}'
     .format(AD.score(X_entrenamiento, y_entrenamiento)))
print('Exactitud de Clasificador AD con set de validacion: {:.2f}'
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

# Gráfica Árbol de Decisión
graph = Source( tree.export_graphviz(AD, out_file=None))
graph.format = 'png'
graph.render('ArbolDecisiones',view=False)

GraficaROC(AD,x,y) # Gráfica ROC modelo AD

# -------------------------- VECINOS K-NN --------------------------------
# -----Entrenamiento
n_neighbors = 2 # Número de vecinos
# MejorNVecinos(range(1, 20),X_entrenamiento, y_entrenamiento,X_evaluacion, y_evaluacion)
 
knn = KNeighborsClassifier(n_neighbors) # Creación clasificador K-NN
Knn = knn.fit(X_entrenamiento, y_entrenamiento) # Entrenamiento del modelo

pred = knn.predict(X_evaluacion) # Predicción instrumentos notas simples
Exactitud_Knn = metrics.accuracy_score(y_evaluacion, pred)*100 # Precisión del modelo en entrenamiento 

print('Exactitud de Clasificador K-NN en set de entrenamiento: {:.2f}'
     .format(knn.score(X_entrenamiento, y_entrenamiento)))
print('Exactitud de Clasificador K-NN en set de validación: {:.2f}'
     .format(knn.score(X_evaluacion, y_evaluacion)))

print(confusion_matrix(y_evaluacion, pred))
print(classification_report(y_evaluacion, pred))

# -----Clasificación
predE = knn.predict(X_ev) # Predicción para los instrumentos mixtos
Exactitud_KnnE = metrics.accuracy_score(y_ev, predE)*100 # Precisión del modelo en evaluación

print('Exactitud de Clasificador K-NN en evaluacion: {:.2f}'
     .format(knn.score(X_ev, y_ev)))

print(confusion_matrix(y_ev, predE))
print(classification_report(y_ev, predE))

GraficaROC(knn,x,y) # Gráfica ROC modelo K-NN

# -------------------------- SVM --------------------------------
# -----Entrenamiento

SVM = SVC(kernel='linear', C=1E10,decision_function_shape='ovr',gamma='auto') # Creación clasificador SVM
SVM = SVM.fit(X_entrenamiento, y_entrenamiento) # Entrenamiento del modelo

vectores=SVM.support_vectors_  # Vectores de soporte
indices=SVM.support_  # Indice de los vectores
nvector=SVM.n_support_ # Número de vectores por cada clase

y_pre= SVM.predict(X_evaluacion) # Predicción instrumentos notas simples
Exactitud_SVM = metrics.accuracy_score(y_evaluacion, y_pre)*100 # Precisión del modelo en entrenamiento 

print('Exactitud de Clasificador SVM para set de entrenamiento: {:.2f}'
     .format(SVM.score(X_entrenamiento, y_entrenamiento)))
print('Exactitud de Clasificador SVM para set de prueba: {:.2f}'
     .format(SVM.score(X_evaluacion, y_evaluacion)))

print(confusion_matrix(y_evaluacion, y_pre))
print(classification_report(y_evaluacion, y_pre))

#-----Clasificación
y_preE= SVM.predict(X_ev) # Predicción para los instrumentos mixtos
Exactitud_SVME = metrics.accuracy_score(y_ev, y_preE)*100 # Precisión del modelo en evaluación

print('Exactitud de Clasificador SVM en evaluacion: {:.2f}'
     .format(SVM.score(X_ev, y_ev)))

print(confusion_matrix(y_ev, y_preE))
print(classification_report(y_ev, y_preE))

GraficaROC(SVM,x,y) # Gráfica ROC modelo SVM



