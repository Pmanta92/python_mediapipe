import cv2
import mediapipe as mp
import sys

# dibuja los puntos de referencia y las conexiones de las manos
mp_drawing = mp.solutions.drawing_utils 

# Inicializa MediaPipe Hands
mp_hands = mp.solutions.hands

# configura los parametros de deteccion de manos
hands = mp_hands.Hands(
    static_image_mode = False,      # Modo de video en tiempo real debe ser "false"
    max_num_hands = 2,              # Dtectar hasta 2 manos
    min_detection_confidence = 0.9, # Umbral de confianza de deteccion
    min_tracking_confidence = 0.8
    )

# Iniciar captura de video
cap = cv2.VideoCapture(0)

while cap.isOpened():   # si la camara esta activada entra al buclw while
    success, camara = cap.read() # leemos la camara / success=bool y image=numpy.ndarray
    image = cv2.resize(camara, (500, 450)) 
    
    # si no hay una conexion exitosa salir
    if not success: 
        print("Ignorando frame vacío.")
        break

    # voltea la imagen horizontalmente - Flip 
    # realiza la conversion  BGR  a  RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # Procesa la imagen y detecta las manos
    results = hands.process(image)

    #print("multi_handedness", results.multi_handedness)    #me muestra (index, score, label)
    #print("multi_hand_landmarks", results.multi_hand_landmarks)    #me indica las coordenadas de los puntos a detectar

    # Convierte la imagen a BGR para OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    Alto, Ancho, _ = image.shape

    # Dibuja las manos detectadas
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            thumb_tip_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            idx_fin_tip_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            # recuperamos las coordenadas de "x, y" pero en froma decimal
            thumb_tip_x = thumb_tip_landmark.x
            thumb_tip_y = thumb_tip_landmark.y
            idx_fin_tip_x = idx_fin_tip_landmark.x
            idy_fin_tip_y = idx_fin_tip_landmark.y
            # transformamos a numeros enteros 
            thumb_tip_x_int = int(thumb_tip_x * Ancho)
            thumb_tip_y_int = int(thumb_tip_y * Alto)
            idx_fin_tip_x_int = int(idx_fin_tip_x * Ancho)
            idx_fin_tip_y_int = int(idy_fin_tip_y * Alto)

            # Dibuja los dos puntos de referencia
            #cv2.circle(image, (thumb_tip_x_int, thumb_tip_y_int), 12, (255,0,0), 5)
            cv2.circle(image, (idx_fin_tip_x_int, idx_fin_tip_y_int), 12, (0,0,255), 10)
            resta_coordenadas = abs(idx_fin_tip_y_int - thumb_tip_y_int)
            #cv2.putText(image, f'Distancia: {resta_coordenadas}', (50, 450),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 5)
            cv2.putText(image, f'X: {idx_fin_tip_x_int} - Y: {idx_fin_tip_y_int}', (20, 400),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 4)

            #print(f"pulgar coordenadas - X: {thumb_tip_x_int}, Y : {thumb_tip_y_int}")
            # mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # mp_drawing.DrawingSpec(color=(0,255,0), thickness=4, circle_radius=7  # Modificar los puntos de referencia
            # mp_drawing.DrawingSpec(color=(255,0,0), thickness=4   # Modificar las lineas de union
            # mp_hands.HAND_CONNECTIONS = dibuja las conexiones de los puntos de referencia

    # Mostrar la imagen
    cv2.imshow('Detector de Manos', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()