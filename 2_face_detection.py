import cv2
import	mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

camaraActivada = cv2.VideoCapture(1)

with mp_face_detection.FaceDetection(
    min_detection_confidence =0.7) as face_detection:

    while True:
        ret, frame = camaraActivada.read()
        if ret == False:
            break

        alto, ancho, _ = frame.shape    # sacamos la dimensiones de la imagen
        frame = cv2.flip(frame, 1)  # volteamos la imagen
        # pasamos a BRG a RGB para trabajar con "mediapipe"
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # aqui tendremos las coordenadas de los 6 puntos claves
        resultado = face_detection.process(frame_rgb)
        #print("Deteccion", resultado.detections)

        if resultado.detections is not None:
            for detection in resultado.detections:
                #---------------- dibuja los 6 puntos -------------------
                #mp_drawing.draw_detection(frame, detection,
                #mp_drawing.DrawingSpec(color=(0,0,255), thickness=10, circle_radius=1),
                #mp_drawing.DrawingSpec(color=(0,255,0), thickness=4))
                #----Bouding Box----
                #print(int(detection.location_data.relative_bounding_box.xmin*ancho))

                #------- Accediendo a los puntos claves, de acuerdo a su nombre -------
                # ojo izquierdo
                x1 = int(detection.location_data.relative_keypoints[0].x*ancho)
                y1 = int(detection.location_data.relative_keypoints[0].y*alto)

                # ojo derecho
                x2 = int(detection.location_data.relative_keypoints[1].x*ancho)
                y2 = int(detection.location_data.relative_keypoints[1].y*alto)

                # nariz
                x3 = int(detection.location_data.relative_keypoints[2].x*ancho)
                y3 = int(detection.location_data.relative_keypoints[2].y*alto)

                # dibujamos un circulo en el rostro
                cv2.circle(frame, (x1, y1), 10, (0,255,0), -1)
                cv2.circle(frame, (x2, y2), 10, (255,0,0), -1)
                cv2.circle(frame, (x3, y3), 10, (0,0,255), -1)
                #---------------------------------------------------------------------

        cv2.imshow("CAMARA", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

camaraActivada.release()
cv2.destroyAllWindows()