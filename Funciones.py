# FUNCIONES

from scipy.fftpack import dct
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelBinarizer 
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from itertools import cycle

import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt

# Funcion que añade ruido a las muestras de entrenamiento 
def adicionruido(audio):
    muestreo, senal_entrada = scipy.io.wavfile.read(audio)  # Lectura audio de entrada
    norma=max(abs(senal_entrada)) # Se normaliza la señal de entrada 
    senalN = senal_entrada/norma
    SNR=-12 #SNR en dB
    
    RMSs=np.sqrt(np.mean((senalN**2)))
    RMSr=RMSs/(10**(SNR/20))
    ruido = np.random.uniform(-RMSr,RMSr,senalN.shape[0])
    senal_ruido=senal_entrada+ruido 
    
     # Se desnormaliza la señal del audio asegurando que siga codificado en 16 bits
    senal_ruidon=senal_ruido*norma  
    maxip=np.int16(32767)     # máxima representación

    if max(abs(senal_ruidon)) >= maxip:  
        norma=np.int16(maxip/max(abs(senal_ruido)))
        senal_ruidon=np.int16(senal_ruido*norma)
    elif max(abs(senal_ruidon)) < maxip:
        senal_ruidon=np.int16(senal_ruidon)
        
    return senal_ruidon, muestreo

def lecturaaudio(audio):
    muestreo, senal_entrada = scipy.io.wavfile.read(audio)  #Lectura audio de entrada
    return senal_entrada,muestreo

#---------------------- Función Factor Mel -------------------
def FactorMel (audio, muestreo):

    #--------------------- Pre enfasis de la señal ----------------------------
    # Filtro de pre enfasis para balancear el espectro de frecuencia
    audio = audio[0:int(3 * muestreo)] # Asegura tomar los 3 primeros segundos  
    pre_enfasis = 0.97
    senal= np.append(audio[0], audio[1:] - pre_enfasis * audio[:-1]) 
      
    #------------------------ Detección de cuadros ----------------------------
    tamano_marco = 0.023 # Tamaño de la ventana en tiempo 
    paso_cuadro = 0.0115 # Tamaño entre cuadros adyacentes en tiempo
    
    long_ventana = tamano_marco*muestreo # longitud de la ventana 
    salto_cuadro = paso_cuadro*muestreo # Número de muestras en el marco actual antes del inicio del siguiente marco
    
    long_senal = len(senal) # longitud de la señal
    long_ventana = int(round(long_ventana)) 
    salto_cuadro = int(round(salto_cuadro))
    
    if long_senal <= long_ventana:
        num_cuadros = 1    # Asegura que se tenga al menos 1 cuadro
    else:
        num_cuadros = 1 + int(np.ceil((1.0*long_senal - long_ventana)/salto_cuadro)) 
    
    pad_longitud_senal = int((num_cuadros - 1)*salto_cuadro + long_ventana)
    
    ceros = np.zeros((pad_longitud_senal - long_senal,))
    # Se asegura que todoslos cuadros tengan igual numero de muestras sin truncarse ninguna muestra de la señal original
    pad_senal = np.concatenate((senal, ceros)) 
    
    aux=np.tile(np.arange(0,num_cuadros*salto_cuadro, salto_cuadro),(long_ventana, 1))
    indices = np.tile(np.arange(0,long_ventana),(num_cuadros,1))+aux.T
    indices = np.array(indices,dtype=np.int32)
    frames = pad_senal[indices]
    
    #------------------------------ Ventaneo ----------------------------------   
    ventaneo=np.hamming(long_ventana) # Se utiliza ventaneo de Hamming
    # frames *= 0.54 - 0.46 * numpy.cos((2 * numpy.pi * n) / (long_ventana - 1))  
    frames *= np.tile(ventaneo,(num_cuadros,1)) 
    
    #--------------------------------- FFT ------------------------------------
    # FFT de N puntos a cada cuadro para calcular el espectro. 
    # Luego se determina el espectro de potencia (Periodograma)
    
    NFFT = 1024
      
    FFT_cuadros = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitud de la FFT para cada cuadro
    potencia_cuadros = 1.0/NFFT * np.square(FFT_cuadros)  # Espectro de potencia para cada cuadro
    
    #------------------------------ Banco de Filtros -------------------------------
    # Banco de filtros triangulares de orden 40 al espectro de potencia para
    # determinar las bandas de frecuancia Mel. 
    
    energia = np.sum(potencia_cuadros,1) # Almacena la energia total en cada cuadro
    energia = np.where(energia == 0,np.finfo(float).eps,energia)
    
    nfilt = 40 # Orden del banco de filtros
    
    frec_baja_mel = 0
    frec_alta_mel = 2595*np.log10(1 + ((muestreo/2)/700))  # Convierte de Hz a Mel la frecuencia alta
    puntosMel = np.linspace(frec_baja_mel, frec_alta_mel, nfilt + 2)  
    # Espaciado igualmente en la escala Mel
    puntosHz = 700*(10**(puntosMel/2595) - 1)  # Convierte Mel a Hz
    bins = np.floor((NFFT+1)*puntosHz/muestreo)
    
    filtro_triangular = np.zeros([nfilt,int(NFFT/2+1)])
    for j in range(0, nfilt):
        f_j_menos = int(bins[j])    # izquierdo
        f_j = int(bins[j+1])        # central
        f_j_mas = int(bins[j+2])    # derecho
    
        for k in range(f_j_menos, f_j):
            filtro_triangular[j,k] = (k - bins[j])/(bins[j+1] - bins[j])
        for k in range(f_j, f_j_mas):
            filtro_triangular[j,k] = (bins[j+2] - k)/(bins[j+2] - bins[j+1])
     
    banco_filtros = np.dot(potencia_cuadros, filtro_triangular.T) # Haya la energia del banco de filtros       
    pot_cuadros= np.where(banco_filtros == 0, np.finfo(float).eps, banco_filtros)  # Estabilidad numérica
    
    banco_filtros = np.log(pot_cuadros)
    
    #---------------------------------- MFCC's --------------------------------
    # Transformada Discreta de Coseno (dct) para descorrelacionar los coeficientes 
    # del filtro bank y obtener su representacion comprimida. 
    
    num_ceps = 13 # Número de coeficientes Mel
    
    mfcc = dct(banco_filtros, type=2, axis=1, norm='ortho')[:,1:num_ceps +1] # Mantiene 1-13
    
    ncuadros, ncoeficientes = np.shape(mfcc)
    n = np.arange(ncoeficientes)
    
    # Se usa un filtro senusoidal en el espectro cepstral para desenfatizar las frecuencias altas
    cep_lifter=22
    
    lift = 1 + (cep_lifter/2) * np.sin(np.pi*n/cep_lifter)
    
    mfcc *= lift 
    mfcc[:,0] = np.log(energia) # Se remplaza el primer coeficiente ceptral con el logaritmo de la energia 
    
    return mfcc


def caracteristicas(coeficientes):
    auxiliar=[]
    Mel=coeficientes.T 
    media = np.mean(coeficientes,axis=0)
    mediana = np.median(coeficientes,axis=0)
    varianza = np.var(coeficientes,axis=0)
    desviacion = np.std(coeficientes,axis=0)
    covarianza =[np.cov(Mel[0,:]),np.cov(Mel[1,:]),np.cov(Mel[2,:]),
                np.cov(Mel[3,:]),np.cov(Mel[4,:]),np.cov(Mel[5,:]),
                np.cov(Mel[6,:]),np.cov(Mel[7,:]),np.cov(Mel[8,:]),np.cov(Mel[9,:]),
                np.cov(Mel[10,:]),np.cov(Mel[11,:]),np.cov(Mel[12,:])]
    datos=[media,mediana,varianza,desviacion,covarianza]
    datos=np.array(datos).T # Arreglo de 13x5
    for k in range(0,len(datos)): # Convierte a un solo vector de 65 características
        auxiliar=np.concatenate((auxiliar, datos[k]), axis=None)
    vect_caracteristicas=auxiliar.T
    
    return vect_caracteristicas


def GraficaROC(Clasificador,x,y):
# ---------------------------- CURVA ROC ------------------------------------
    
    y=LabelBinarizer().fit_transform(y) # Se binarizan las clases 
    nclases = 4 # Cantidad de clases (4 instrumentos)
    
    X_entranamiento, X_evaluacion, y_entranamiento, y_evaluaciona = train_test_split(x, y, test_size=.3,random_state=42)
    # Se hace la predección de cada clase contra las otras
    clas = OneVsRestClassifier(Clasificador)
    
    y_validacion = clas.fit(X_entranamiento, y_entranamiento).predict(X_evaluacion)
    
    # Cálculo de la curva ROC y el area bajo la ROC para cada clase
    fpr = dict()#Tasa de falsos positivos
    tpr = dict() #Tasa de Verdaderos positivos
    roc_auc = dict() #Area bajo la curva (ABC)
    
    for i in range(nclases):
        fpr[i], tpr[i], _ = roc_curve(y_evaluaciona[:, i], y_validacion[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Se agregan todos los valores de los falsos positivos de cada clase
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(nclases)]))
    
    # Interpolación de toda las curvas ROC para cada clase en el punto ideal
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(nclases):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])   
    
    # Se promedia y se computa el ABC
    mean_tpr /= nclases
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Gráfica de las curvas ROC de todas las clases 
    plt.figure()
    lw=2
    plt.plot(fpr["macro"], tpr["macro"],
             label='Curva ROC macro-promedio (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='red', linestyle=':', linewidth=4)
    colors = cycle(['skyblue', 'darkorange', 'mediumpurple','greenyellow'])
    for i, color in zip(range(nclases), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                  label='Curva ROC clase {0} (area = {1:0.2f})'
                  ''.format(i, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa Falsos Positivos')
    plt.ylabel('Tasa Verdaderos Positivos')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show()
    
    return

def MejorNVecinos(k_range,X_entranamiento, y_entranamiento,X_evaluacion, y_evaluacion):
    scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors = k)
        knn.fit(X_entranamiento, y_entranamiento)
        scores.append(knn.score(X_evaluacion, y_evaluacion))
    plt.figure()
    plt.xlabel('k')
    plt.ylabel('Exactitud')
    plt.scatter(k_range, scores)
    plt.xticks([0,2,4,6,8,10,12,14,16,18,20])
    
    return
