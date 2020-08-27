
import numpy as np
import scipy.io.wavfile
from scipy.fftpack import dct
import decimal
import matplotlib.pyplot as plt


def round_half_up(number):
    return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))

# Funcion que añade ruido a las muestras de entrenamiento 
def adicionruido(audio):
    muestreo, senal_entrada = scipy.io.wavfile.read(audio)  #Lectura audio de entrada
    norma=max(abs(senal_entrada))
    senalN = senal_entrada/norma
    SNR=-3
    
    RMSs=np.sqrt(np.mean((senalN**2)/2))
    RMSr=RMSs/(10**(SNR/20))
    ruido = np.random.uniform(-RMSr,RMSr,senalN.shape[0])
    senal_ruido=senal_entrada+ruido 
    return senal_ruido, muestreo

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
    datos=np.array(datos).T
    for k in range(0,len(datos)):
        auxiliar=np.concatenate((auxiliar, datos[k]), axis=None)
    datos=auxiliar.T
    return datos

#--------------------- Pre enfasis de la señal ----------------------------
# Filtro de pre enfasis para balancear el espectro de frecuencia, 
# normalmente altas frecuencias tienen menor magnitud que las frecuencias bajas.
# Tambien mejora la relación señal a ruido SNR.
senal_entrada, muestreo=adicionruido("clarinete_notas1.wav")
pre_enfasis = 0.97
senal= np.append(senal_entrada[0], senal_entrada[1:] - pre_enfasis * senal_entrada[:-1])
  
#------------------------ Detección de cuadros ----------------------------
# Las frecuencias en la muestra van cambiando en el tiempo por lo que se busca
# una buena aproximación de los contornos de la frecuancia en periodos cortos 
# de tiempo concatenando cuadros adjacentes.

tamano_marco = 0.023 #Tamaño de la ventana en tiempo 
paso_cuadro = 0.0115 #Tamaño de cuadro entre cuadros

long_ventana = tamano_marco*muestreo # longitud de la ventana 
salto_cuadro = paso_cuadro*muestreo # Número de muestras en el marco actual antes del inicio del siguiente marco

long_senal = len(senal) #longitud de la señal
long_ventana = int(round_half_up(long_ventana))
salto_cuadro = int(round_half_up(salto_cuadro))

if long_senal <= long_ventana:
    num_cuadros = 1    # Asegura que se tenga al menos 1 cuadro
else:
    num_cuadros = 1 + int(np.ceil((1.0*long_senal - long_ventana)/salto_cuadro)) 

pad_longitud_senal = int((num_cuadros - 1)*salto_cuadro + long_ventana)

ceros = np.zeros((pad_longitud_senal - long_senal,))
pad_senal = np.concatenate((senal, ceros)) #Para asegurar que todos los cuadros tengan igual numero de muestras sin truncarse ninguna muestra de la señal original

indices = np.tile(np.arange(0, long_ventana), (num_cuadros,1))+np.tile(np.arange(0,num_cuadros*salto_cuadro, salto_cuadro), (long_ventana, 1)).T
indices = np.array(indices,dtype=np.int32)
frames = pad_senal[indices]

#------------------------------ Ventaneo ----------------------------------
# Se implementa para contrarrestar los efectos de borde al segmentar la señal 
# en  cuadros y asi reducir la fuga espectral por los cambios repentinos de 
# cero a señal y de señal a cero.

ventaneo=np.hamming(long_ventana) # Se utiliza ventaneo de Hamming
# frames *= 0.54 - 0.46 * numpy.cos((2 * numpy.pi * n) / (long_ventana - 1))  
frames *= np.tile(ventaneo,(num_cuadros,1)) 
plt.plot(ventaneo)
plt.title('Ventana Hamming')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.show()

#--------------------------------- FFT ------------------------------------
# Se realiza la FFT de N puntos a cada cuadro para calcular el espectro. 
# Luego se determina el espectro de potencia (Periodograma)

NFFT = 1024

# comienzo_tiempo= time.time()
FFT_cuadros = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitud de la FFT para cada cuadro
potencia_cuadros = 1.0/NFFT * np.square(FFT_cuadros)  # Espectro de potencia para cada cuadro
# fin_tiempo = time.time()
# tiempoFFT=fin_tiempo-comienzo_tiempo
# print(tiempoFFT)

#------------------------------ Filtro Bank -------------------------------
# Se implementa un filtro triangular de orden 40 al espectro de potencia para
# determinar las bandas de frecuancia Mel. 

energia = np.sum(potencia_cuadros,1) # Almacena la energia total en cada cuadro
energia = np.where(energia == 0,np.finfo(float).eps,energia)

nfilt = 40

frec_baja_mel = 0
frec_alta_mel = 2595*np.log10(1 + ((muestreo/2)/700))  # Convierte de Hz a Mel la frecuencia alta
puntosMel = np.linspace(frec_baja_mel, frec_alta_mel, nfilt + 2)  
# Espaciado igualmente en la escala Mel
puntosHz = 700*(10**(puntosMel/2595) - 1)  # Convierte Mel a Hz
bins = np.floor((NFFT+1)*puntosHz/muestreo)

fbank = np.zeros([nfilt,int(NFFT/2+1)])

 
for j in range(0, nfilt):
    f_j_menos = int(bins[j-1])  # izquierdo
    f_j = int(bins[j])          # central
    f_j_mas = int(bins[j+1])    # derecho

    for k in range(f_j_menos, f_j):
        fbank[j-1,k] = (k - bins[j-1])/(bins[j] - bins[j-1])
    for k in range(f_j, f_j_mas):
        fbank[j-1,k] = (bins[j+1] - k)/(bins[j+1] - bins[j])

plt.plot(fbank)
plt.title('Filtro Bank')
plt.xlabel('Frecuencia')
plt.ylabel('Amplitud')
plt.show()
filter_banks = np.dot(potencia_cuadros, fbank.T) # Haya la energia del filtro bank       
pot_cuadros= np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Estabilidad numérica

filter_banks = np.log(pot_cuadros)

#---------------------------------- MFCC's --------------------------------
# Los coeficientes del filtro bank estan muy correlacionados por lo que se 
# debe aplicar una Transformada Discreta de Coseno (dct) para descorrelacionarlos 
# y ceder a una representacion comprimida del filtro Bank. Para el reconocimiento 
# de audio o de voz automatico se toman por lo general los primeros 12 coeficientes
# ceptrales.

num_ceps = 13

mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:,1:num_ceps +1] # Mantiene 1-13

ncuadros, ncoeficientes = np.shape(mfcc)
n = np.arange(ncoeficientes)

cep_lifter=22

lift = 1 + (cep_lifter/2) * np.sin(np.pi*n/cep_lifter)
# Se usa un filtro senusoidal en el espectro cepstral para desenfatizar las frecuencias altas
mfcc *= lift 
mfcc[:,0] = np.log(energia) #remplaza el primer coeficiente ceptral con el logaritmo de la energia del cuadro 
    
             
estadisticas=caracteristicas(mfcc)
Instrumento=[0]


        

# plt.plot(noise)
# plt.title('Ruido')
# plt.show()
# plt.plot(muestra)
# plt.title('Senal con ruido')
# plt.show()

# scipy.io.wavfile.write('Clarineteruido.wav',muestreo,muestra)










