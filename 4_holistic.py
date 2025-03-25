import cv2
import	mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

camaraActivada = cv2.VideoCapture("ejercicio2.mp4")

with mp_holistic.Holistic(
    static_image_mode = False,
    model_complexity = 1,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.9) as holistic:

    while True:
        ret, frame = camaraActivada.read()
        if ret == False:
            break

        # Corrige la rotación si es necesario
        #qframe = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame = cv2.resize(frame, (450,500))

        alto, ancho, _ = frame.shape    # sacamos la dimensiones de la imagen
        frame = cv2.flip(frame, 1)  # volteamos la imagen
        # pasamos a BRG a RGB para trabajar con "mediapipe"
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # aqui tendremos las "pose_landmarks, pose_world_landmarks, 
        # face_landmarks, left_hand_landmarks, right_hand_landmarks"
        resultado = holistic.process(frame_rgb)
        #print(resultado.right_hand_landmarks)
        mp_drawing.draw_landmarks(frame, resultado.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0,0,255), thickness=5, circle_radius=1),
        mp_drawing.DrawingSpec(color=(255,255,0), thickness=3))

        mp_drawing.draw_landmarks(frame, resultado.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0,0,255), thickness=5, circle_radius=1),
        mp_drawing.DrawingSpec(color=(255,255,0), thickness=3))

        mp_drawing.draw_landmarks(frame, resultado.pose_landmarks,mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(255,0,0), thickness=5, circle_radius=1),
        mp_drawing.DrawingSpec(color=(255,255,0), thickness=3))

        cv2.imshow("CAMARA", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

camaraActivada.release()
cv2.destroyAllWindows()