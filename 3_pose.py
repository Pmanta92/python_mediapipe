import cv2
import	mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

camaraActivada = cv2.VideoCapture("ejercicio1.mp4")

with mp_pose.Pose(
    static_image_mode = False,
    model_complexity = 1,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5) as pose:

    while True:
        ret, frame = camaraActivada.read()
        if ret == False:
            break
        # Corrige la rotación si es necesario
        #frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame = cv2.resize(frame, (600,650))

        alto, ancho, _ = frame.shape    # sacamos la dimensiones de la imagen
        frame = cv2.flip(frame, 1)  # volteamos la imagen
        # pasamos a BRG a RGB para trabajar con "mediapipe"
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # aqui tendremos las coordenadas de los 33 puntos"pose_landmarks"
        resultado = pose.process(frame_rgb)
        #print("Pose landmarks", resultado.pose_landmarks)

        if resultado.pose_landmarks is not None:
            #---------------- dibuja los 33 puntos -------------------
            #mp_drawing.draw_landmarks(frame, resultado.pose_landmarks,mp_pose.POSE_CONNECTIONS,
            #mp_drawing.DrawingSpec(color=(0,0,255), thickness=10, circle_radius=1),
            #mp_drawing.DrawingSpec(color=(255,255,0), thickness=8))

            #------- Accediendo a los puntos claves, de acuerdo a su nombre -------
            #print(int(resultado.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x*ancho))
            # HOMBRO IZQUIERDO
            x1 = int(resultado.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x*ancho)
            y1 = int(resultado.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y*alto)
            # CODO IZQUIERDO
            x2 = int(resultado.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x*ancho)
            y2 = int(resultado.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y*alto)
            # CODO IZQUIERDO
            x3 = int(resultado.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x*ancho)
            y3 = int(resultado.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y*alto)

            # dibujamos un circulo en el rostro
            cv2.line(frame, (x1, y1), (x2, y2), (0,255,255), 6)
            cv2.line(frame, (x2, y2), (x3, y3), (0,255,255), 6)
            cv2.circle(frame, (x1, y1), 10, (0,255,0), -1)
            cv2.circle(frame, (x2, y2), 10, (0,255,0), -1)
            cv2.circle(frame, (x3, y3), 10, (0,255,0), -1)
            cv2.putText(frame, f'Hombro: x1={x1}/y1={y1}', (10, 550),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(frame, f'Codo: x2={x2}/y2={y2}', (10, 575),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(frame, f'Muñeca: x3={x3}/y3={y3}', (10, 600),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0 ), 2)

        cv2.imshow("CAMARA", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

camaraActivada.release()
cv2.destroyAllWindows()