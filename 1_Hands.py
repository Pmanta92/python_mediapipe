import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

camaraActivada = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode = False,
    max_num_hands = 2,
    min_detection_confidence = 0.5) as hands:

    while True:
        ret, frame = camaraActivada.read()
        if ret == False:
            break

        alto, ancho, _ = frame.shape    # sacamos la dimensiones de la imagen
        frame = cv2.flip(frame, 1)  # volteamos la imagen
        # pasamos a BRG a RGB para trabajar con "mediapipe"
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # aqui tendremos dos salidas "multi_handedness y multi_hand_landmarks"
        resultado = hands.process(frame_rgb)
        #print(resultado.multi_handedness)   # nos entrega "index, score, label"
        #print(resultado.multi_hand_landmarks)   # nos entrega las coordenadas de los 21 puntos en decimales

        if resultado.multi_hand_landmarks is not None:  # coondision solo para cuando detecte manos
            #----------------------------------------------------------------------------------------------
            for hand_landmarks in resultado.multi_hand_landmarks:
                #---------------- dibuja los 21 puntos -------------------
                #mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                #                          mp_drawing.DrawingSpec(color=(0,0,255), thickness=4, circle_radius=7),
                #                          mp_drawing.DrawingSpec(color=(0,255,0), thickness=4))

                #------- Accediendo a los puntos claves, de acuerdo a su nombre -------
                x1 = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x*ancho)
                y1 = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y*alto)

                x2 = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x*ancho)
                y2 = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y*alto)

                x3 = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x*ancho)
                y3 = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y*alto)

                x4 = int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x*ancho)
                y4 = int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y*alto)

                x5 = int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x*ancho)
                y5 = int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y*alto)
                # dibujamos un circulo en cada dedo
                cv2.circle(frame, (x1, y1), 15, (255,0,0), -1)
                cv2.circle(frame, (x2, y2), 15, (0,255,0), -1)
                cv2.circle(frame, (x3, y3), 15, (0,0,255), -1)
                cv2.circle(frame, (x4, y4), 15, (255,0,255), -1)
                cv2.circle(frame, (x5, y5), 15, (255,255,0), -1)
                #---------------------------------------------------------------------

        cv2.imshow("CAMARA", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

camaraActivada.release()
cv2.destroyAllWindows()