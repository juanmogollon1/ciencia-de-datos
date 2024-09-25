import cv2
import numpy as np
from tensorflow.keras.models import load_model
#import RPi.GPIO as GPIO  # Para controlar el servomotor

# Configurar el GPIO para el servomotor
GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.OUT)
servo = GPIO.PWM(17, 50)  # Configurar el pin 17 para PWM a 50Hz
servo.start(0)  # Iniciar el servomotor en 0 grados

# Cargar el modelo
modelo_cargado = load_model("modelo_deteccion.h5")

# Cargar el clasificador de Haar Cascades para detección de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Iniciar la captura de video
cap = cv2.VideoCapture(0)
umbral_confianza = 0.5  # Ajusta este valor según sea necesario

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar el frame.")
        break
    
    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detectar rostros
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        # Si no se detecta ningún rostro, mostrar "Oculta"
        cv2.putText(frame, "Oculta", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        for (x, y, w, h) in faces:
            # Ampliar la caja alrededor de la cara
            x -= 20  # Ajusta este valor para ampliar horizontalmente
            y -= 20  # Ajusta este valor para ampliar verticalmente
            w += 40   # Ajusta este valor para ampliar horizontalmente
            h += 40   # Ajusta este valor para ampliar verticalmente

            # Asegúrate de que las coordenadas no salgan del marco
            x = max(x, 0)
            y = max(y, 0)
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Extraer la región de interés (ROI) del rostro
            roi = frame[y:y + h, x:x + w]
            roi_resized = cv2.resize(roi, (224, 224)) / 255.0  # Cambiado a 224x224
            roi_array = np.expand_dims(roi_resized, axis=0)

            # Hacer predicción
            prediccion = modelo_cargado.predict(roi_array)

            # Imprimir predicción
            print("Predicción:", prediccion[0][0])  # Ver la confianza

            if prediccion[0][0] > umbral_confianza:
                etiqueta = "Destapada"
                # Activar el servomotor cuando la cara esté destapada
               # servo.ChangeDutyCycle(7.5)  # Mueve el servomotor a 90 grados
            else:
                etiqueta = "Oculta"
                # Mover el servomotor de vuelta cuando la cara esté oculta
               # servo.ChangeDutyCycle(2.5)  # Mueve el servomotor a 0 grados

            cv2.putText(frame, etiqueta, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Mostrar el frame con las predicciones
    cv2.imshow('Captura de Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
servo.stop()
GPIO.cleanup()  # Limpiar los pines GPIO al terminar
