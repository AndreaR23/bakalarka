import cv2

image = cv2.imread('/home/ajuska/Plocha/bakalarka/silv.jpg')
face_classif = cv2.CascadeClassifier('/home/ajuska/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_default.xml')
eye_classif = cv2.CascadeClassifier('/home/ajuska/opencv-3.1.0/data/haarcascades/haarcascade_eye.xml')

grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = face_classif.detectMultiScale(image, 1.3, 5)

for (x,y,w,h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_gray = grey[y:y + h, x:x + w]
    roi_color = image[y:y + h, x:x + w]
    eyes = eye_classif.detectMultiScale(roi_gray)
    for(ex,ey,ew,eh) in eyes:
       cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

cv2.imshow('foto', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


