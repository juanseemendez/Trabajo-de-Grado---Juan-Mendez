import time
import joblib
import paho.mqtt.client as mqtt
from Funciones import lecturaaudio
from Funciones import FactorMel
from Funciones import caracteristicas
  
comienzo_tiempo= time.time() # Comienzo medición de tiempo

SVM = joblib.load('SVMFinal.pkl') # Se carga el modelo entrenado

senal_entrada, muestreo=lecturaaudio('Mixto57.wav') # lectura audio entrada 
mfccout= FactorMel(senal_entrada, muestreo) # Se extraen los factores Mel
X_ev=caracteristicas(mfccout) # Extracción de medidas estadísticas
X_ev=X_ev.reshape(1,-1)

y_pre= SVM.predict(X_ev) # Predicción del instrumento   

if y_pre == 0: # Asignación nombre de acuerdo a la clase
    Instrumento='Clarinete'
elif y_pre == 1:
    Instrumento='Flauta'
elif y_pre == 2:
    Instrumento='Trombon'
elif y_pre == 3:
    Instrumento='Timbal'
    
def on_connect(client, userdata, flags, rc): # Determina estado de conexión
    if rc == 0: 
        global conectado
        conectado=True
    else:
        conectado=False 
    
conectado = False 
broker="test.mosquitto.org"
puerto=1883 

client = mqtt.Client(client_id='Publicacion', clean_session=False) # Cliente como publicador
client.on_connect = on_connect
client.connect(broker,port=puerto) # Se conecta al broker y al puerto
client.loop_start()     # Inicio Transmisión 
client.publish("MFCC/clasificacion","Instrumento clasificado es: "+ str(Instrumento)) # Publicación
client.loop_stop() 
client.disconnect()     # Fin Comunicación  
 
fin_tiempo = time.time() # Fin tiempo empleado 
tiempo_sistema=fin_tiempo-comienzo_tiempo
print(tiempo_sistema) 




