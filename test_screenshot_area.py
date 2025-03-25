import cv2
import numpy as np
import pyautogui

while True:
    screnshot = pyautogui.screenshot(region= (245, 160, 640, 400))  #X, Y, ANCHO, ALTO
    screnshot = np.array(screnshot)
    screnshot = cv2.cvtColor(screnshot, cv2.COLOR_BGR2RGB)  # pasamos de BGR a RGB

    cv2.imshow("RECORTE PANTALLA", screnshot)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()