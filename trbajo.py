import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Rutas de acceso
ruta_destapado = "C:\\Users\\Edward\\Downloads\\Destapado"
ruta_oculto = "C:\\Users\\Edward\\Downloads\\Oculto"

# Listas para almacenar imágenes y etiquetas
imagenes = []
etiquetas = []

# Cargar imágenes destapadas
for archivo in os.listdir(ruta_destapado):
    if archivo.endswith(('.png', '.jpg', '.jpeg')):
        img = cv2.imread(os.path.join(ruta_destapado, archivo))
        img = cv2.resize(img, (224, 224))  # Redimensionar
        imagenes.append(img)
        etiquetas.append(1)  # 1 para destapada

# Cargar imágenes ocultas
for archivo in os.listdir(ruta_oculto):
    if archivo.endswith(('.png', '.jpg', '.jpeg')):
        img = cv2.imread(os.path.join(ruta_oculto, archivo))
        img = cv2.resize(img, (224, 224))  # Redimensionar
        imagenes.append(img)
        etiquetas.append(0)  # 0 para oculta

# Convertir listas a arrays
imagenes = np.array(imagenes) / 255.0  # Normalizar
etiquetas = np.array(etiquetas)

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(imagenes, etiquetas, test_size=0.2, random_state=42)

# Construir el modelo
modelo = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Salida binaria
])

# Compilar el modelo
modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
modelo.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Evaluar el modelo
pérdida, precisión = modelo.evaluate(X_test, y_test)
print(f'Precisión del modelo: {precisión:.2f}')

# Guardar el modelo
modelo.save("modelo_deteccion.h5")
