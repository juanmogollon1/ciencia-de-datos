import time
import cv2
import os
import numpy as np

# Rutas a las carpetas de imágenes "Oculto" y "Destapado"
dataPath_oculto = r"C:\Users\Edward\Downloads\Oculto"
dataPath_destapado = r"C:\Users\Edward\Downloads\Destapado"

# Listas para almacenar las imágenes y sus respectivas etiquetas
labels = []
facesData = []
label = 0  # 0 para "Oculto", 1 para "Destapado"

# Definir el tamaño al que deseas redimensionar las imágenes (ejemplo: 100x100 píxeles)
image_size = (100, 100)

# Cargar imágenes de la categoría "Oculto"
for file_name in os.listdir(dataPath_oculto):
    image_path = os.path.join(dataPath_oculto, file_name)
    print("Procesando imagen Oculto:", image_path)
    
    # Leer imagen en escala de grises
    image = cv2.imread(image_path, 0)
    
    if image is None:
        print(f"Error al cargar la imagen: {image_path}")
        continue
    
    # Redimensionar la imagen al tamaño definido
    image_resized = cv2.resize(image, image_size)
    
    facesData.append(image_resized)
    labels.append(0)

# Cargar imágenes de la categoría "Destapado"
for file_name in os.listdir(dataPath_destapado):
    image_path = os.path.join(dataPath_destapado, file_name)
    print("Procesando imagen Destapado:", image_path)
    
    # Leer imagen en escala de grises
    image = cv2.imread(image_path, 0)
    
    if image is None:
        print(f"Error al cargar la imagen: {image_path}")
        continue
    
    # Redimensionar la imagen al tamaño definido
    image_resized = cv2.resize(image, image_size)
    
    facesData.append(image_resized)
    labels.append(1)

# Imprimir la cantidad de imágenes por etiqueta
print("Cantidad de imágenes en 'Oculto':", np.count_nonzero(np.array(labels) == 0))
print("Cantidad de imágenes en 'Destapado':", np.count_nonzero(np.array(labels) == 1))

# Crear el reconocedor de rostros usando LBPH
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Temporizador de inicio
start_time = time.time()

# Entrenar el modelo (una sola vez)
print("Entrenando el modelo...")
face_recognizer.train(facesData, np.array(labels))

# Temporizador de fin
end_time = time.time()

# Almacenar el modelo entrenado
face_recognizer.write("face_mask_model.xml")
print("Modelo almacenado exitosamente.")

# Mostrar tiempo total de entrenamiento
training_time = end_time - start_time
print(f"Tiempo total de entrenamiento: {training_time} segundos")


