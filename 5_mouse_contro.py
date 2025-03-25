import cv2
import mediapipe as mp
import numpy as np
import pyautogui

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

camaraActivada = cv2.VideoCapture(1)
color_mause_pointer = (255, 0, 0)

# Puntos de interes del juego
screen_game_x_ini = 245
screen_game_y_ini = 160
screen_game_x_fin = 245 + 400
screen_game_y_fin = 160 + 640

aspect_ratio_screen = (screen_game_x_fin - screen_game_x_ini) / (screen_game_y_fin - screen_game_y_ini)
print(aspect_ratio_screen)

x_y_ini = 50

def calcular_distancia(x1, y1, x2, y2):
    p1 = np.array([x1, y1])
    p2 = np.array([x2, y2])
    return np.linalg.norm(p1 - p2)

def detect_finger_down(hand_landmarks):
    finger_down = False
    color_base = (255, 0, 0)
    color_index = (255, 100, 255)

    x_base1 = int(hand_landmarks.landmark[5].x * ancho)
    y_base1 = int(hand_landmarks.landmark[5].y * alto)

    x_base2 = int(hand_landmarks.landmark[6].x * ancho)
    y_base2 = int(hand_landmarks.landmark[6].y * alto)

    x_index = int(hand_landmarks.landmark[4].x * ancho)
    y_index = int(hand_landmarks.landmark[4].y * alto)

    d_base = calcular_distancia(x_base1, y_base1, x_base2, y_base2)
    d_base_index = calcular_distancia(x_base1, y_base1, x_index, y_index)
    cv2.putText(output, f'condicion: {int(d_base_index)} < {int(d_base)}', (10, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)

    if d_base_index < d_base:
        finger_down = True
        color_base = (0, 0, 255)
        color_index = (0, 0, 255)

    cv2.circle(output, (x_base1, y_base1), 10, color_base , 3)
    cv2.circle(output, (x_index, y_index), 10, color_index , 3)
    cv2.line(output, (x_base1, y_base1), (x_base2, y_base2), color_base, 3)
    cv2.line(output, (x_base1, y_base1), (x_index, y_index), color_index, 3)
    return finger_down


with mp_hands.Hands(
    static_image_mode = False,
    max_num_hands = 1,  # deteccion solo una mano
    min_detection_confidence = 0.25) as hands:

    while True:
        ret, frame = camaraActivada.read()
        if ret == False:
            break

        alto, ancho, _ = frame.shape    # sacamos la dimensiones de la imagen
        frame = cv2.flip(frame, 1)  # volteamos la imagen

        # dibujando un area proporcional a la del juego
        area_ancho = ancho - x_y_ini * 2
        area_alto = int(area_ancho / aspect_ratio_screen)
        aux_image = np.zeros(frame.shape, np.uint8)
        aux_image = cv2.rectangle(aux_image, (x_y_ini, x_y_ini), (x_y_ini + area_ancho, x_y_ini + area_alto), (0, 255, 0), 5)
        output = cv2.addWeighted(frame, 1, aux_image, 0.7, 0)
        # pasamos a BRG a RGB para trabajar con "mediapipe"
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # aqui tendremos dos salidas "multi_handedness y multi_hand_landmarks"
        resultado = hands.process(frame_rgb)

        if resultado.multi_hand_landmarks is not None:  # coondision solo para cuando detecte manos
            #----------------------------------------------------------------------------------------------
            for hand_landmarks in resultado.multi_hand_landmarks:
                #--------------- dibuja el punto de referencia -----------------
                x = int(hand_landmarks.landmark[6].x * ancho)
                y = int(hand_landmarks.landmark[6].y * alto)
                xm = np.interp(x, (x_y_ini, x_y_ini + area_ancho), (screen_game_y_ini, screen_game_y_fin))
                ym = np.interp(y, (x_y_ini, x_y_ini + area_alto), (screen_game_y_ini, screen_game_y_fin)) 
                pyautogui.moveTo(int(xm), int(ym))
                
                if detect_finger_down(hand_landmarks):
                    pyautogui.click()
                
                cv2.circle(output, (x, y), 15, color_mause_pointer , 3)
                cv2.circle(output, (x, y), 10, color_mause_pointer , -1)

        #cv2.imshow("CAMARA", frame)
        cv2.imshow("output", output)
        if cv2.waitKey(1) & 0xFF == 27:
            break

camaraActivada.release()
cv2.destroyAllWindows()