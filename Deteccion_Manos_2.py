import cv2
import mediapipe as mp
import serial, time

ser = serial.Serial("COM6", 9600, write_timeout= 10)

dsipositivoCaptura = cv2.VideoCapture(0)    # llamamos la camara

mpManos = mp.solutions.hands    # ingresa los diferentes puntos de la mano

manos = mpManos.Hands(static_image_mode = False,        # no usaremos imagenes estatica
                      max_num_hands = 2,                # numeros maximos de manos
                      min_detection_confidence = 0.9,   # minimo de confianza de deteccion 
                      min_tracking_confidence = 0.8)    # la confianza de seguimiento de las manos

mpDibujar = mp.solutions.drawing_utils  # sirve para dibujar dentro del cuadro de la camara

while dsipositivoCaptura.isOpened():    # si la camara esta activada entra al buclw while
    succes, camara = dsipositivoCaptura.read()  # leemos la camara
    img = cv2.resize(camara, (600, 500))    # dimensionamos la pantalla 
    imgRGB = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)   #pasamos de BGR a RGB para trabajar con la libreria "mediapipe"
    resultado = manos.process(imgRGB)   
    img = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2BGR)  # volvemos a pasar a RGB a BGR para Opencv         

    if resultado.multi_hand_landmarks:
        for handLms in resultado.multi_hand_landmarks:  #accedemos a las coordenadas de los puntos de la mano
            mpDibujar.draw_landmarks(img, handLms, mpManos.HAND_CONNECTIONS, mpDibujar.DrawingSpec(color=(255,0,0), thickness = 4, circle_radius=5),
                                                                             mpDibujar.DrawingSpec(color=(0,0,0), thickness = 6))    #dibuja las conexiones con los puntos de la mano(opcinal)
            for id, lm in enumerate(handLms.landmark):
                alto, ancho, color = img.shape  # sacamos laS dimensiones de la pantalla
                cx, cy = int(lm.x*ancho), int(lm.y*alto)    #multiplicamos las dimensiones de la pantalla para saber las coordenadas
                cxx, cyy = int(lm.x*180), int(lm.y*180) # convertimos de 0 a 180 grados para los servo

                #if (id == 20 or id == 4) and handedness=="Left": # Mano izquierda
                #    cv2.circle(img, (cx, cy), 15, (255,0,0), cv2.FILLED)
                #    ser.write(f"{cxx},{cyy}".encode())  # Convierte la variable a cadena, luego a bytes
                
                if (id == 20) and handedness =="Right":   # Mano derecha
                    cv2.circle(img, (cx, cy), 15, (0,0,255), cv2.FILLED)
                    ser.write(f"{cxx},{cyy}".encode())  # Convierte la variable a cadena, luego a bytes
                
                for hand_info in resultado.multi_handedness: # guardamos (index, score, label)
                    handedness = hand_info.classification[0].label
                    score = hand_info.classification[0].score

    # Dibujamos las coordenadas X, Y            
    cv2.arrowedLine(img, (25,25), (545,25), (0,0,255), 5, tipLength = 0.04)
    cv2.arrowedLine(img, (25,25), (25,440), (0,255,0), 5, tipLength = 0.04)
    cv2.putText(img, "X", (550, 43),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 5)
    cv2.putText(img, "Y", (12, 480),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 5)
    # Mostrar la imagen
    cv2.imshow('Detector de Manos', img)
    if cv2.waitKey(5) & 0xFF == 27:
        break
#muestra las posiciones de los diferentes puntos de la mano y me muestra (index, score, label)
print(f"multi_hand_landmarks: {resultado.multi_hand_landmarks} / multi_handedness: {resultado.multi_handedness}") 
ser.close()
dsipositivoCaptura.release()
cv2.destroyAllWindows()

