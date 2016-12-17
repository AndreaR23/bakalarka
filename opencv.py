import numpy as np
import cv2
from picamera import PiCamera
from time import sleep

kamera = PiCamera()

kamera.start_preview()
sleep(3)
kamera.capture('/home/pi/Desktop/foto.jpg')
sleep(2)
kamera.stop_preview()

fotka = cv2.imread('/home/pi/Desktop/foto.jpg')

oblicej_klas = cv2.CascadeClassifier('/home/pi/opencv-2.4.10/data/haarcascades/haarcascade_frontalface_default.xml')
oko_klas = cv2.CascadeClassifier('/home/pi/opencv-2.4.10/data/haarcascades/haarcascade_eye.xml')

seda = cv2.cvtColor(fotka,cv2.COLOR_BGR2GRAY)

obliceje = oblicej_klas.detectMultiScale(fotka, 1.3, 5)

for (x,y,w,h) in obliceje:
    cv2.rectangle(fotka,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = seda[y:y+h, x:x+w]
    roi_color = fotka[y:y+h, x:x+w]
    oci = oko_klas.detectMultiScale(roi_gray)
    for(ex,ey,ew,eh) in oci:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

cv2.imshow('foto',fotka)
cv2.waitKey(0)
cv2.destroyAllWindows()
