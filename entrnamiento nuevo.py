import cv2
import  mediapipe as mp



# Inicializar el detector de rostros de mediapipe
mp_face_detection = mp.solutions.face_detection

# Etiquetas de predicción para el modelo
LABELS = ["Oculto", "Destapado"]

# Cargar el modelo LBPH entrenado
face_mask = cv2.face.LBPHFaceRecognizer_create()
face_mask.read("face_mask_model.xml")

# Inicializar la captura de video (cámara)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Configuración de la detección facial con mediapipe
with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Voltear la imagen para efecto espejo
        frame = cv2.flip(frame, 1)

        # Obtener las dimensiones del frame
        height, width, _ = frame.shape

        # Convertir el frame a RGB para la detección facial con mediapipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesar el frame para la detección de rostros
        results = face_detection.process(frame_rgb)

        if results.detections is not None:
            for detection in results.detections:
                # Extraer las coordenadas del bounding box
                bbox = detection.location_data.relative_bounding_box
                xmin = int(bbox.xmin * width)
                ymin = int(bbox.ymin * height)
                w = int(bbox.width * width)
                h = int(bbox.height * height)

                # Asegurarse de que los valores sean válidos
                if xmin < 0 or ymin < 0 or (xmin + w) > width or (ymin + h) > height:
                    continue

                # Recortar la imagen de la cara detectada
                face_image = frame[ymin:ymin + h, xmin:xmin + w]
                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                face_image = cv2.resize(face_image, (72, 72), interpolation=cv2.INTER_CUBIC)

                # Predecir si es "Oculto" o "Destapado"
                result = face_mask.predict(face_image)

                # Mostrar el resultado en la imagen si la predicción es confiable
                if result[1] < 150:  # Umbral para decidir si la predicción es confiable
                    label = LABELS[result[0]]
                    color = (0, 255, 0) if label == "Destapado" else (0, 0, 255)
                    
                    # Dibujar el texto y el rectángulo alrededor de la cara
                    cv2.putText(frame, "{}".format(label), (xmin, ymin - 15), 2, 1, color, 2, cv2.LINE_AA)
                    cv2.rectangle(frame, (xmin, ymin), (xmin + w, ymin + h), color, 2)

        # Mostrar el frame con las predicciones
        cv2.imshow("Frame", frame)

        # Presionar 'Esc' para salir del bucle
        if cv2.waitKey(1) & 0xFF == 27:
            break

# Liberar la captura y cerrar ventanas
cap.release()
cv2.destroyAllWindows()
