from keras.models import load_model
from knihovna import zm_vel
import numpy as np
import cv2
import os
from matplotlib import pyplot


l_sablony = []
l_sablony = zm_vel('/home/ajuska/Plocha/bakalarka/oci/oko1.jpg', 5, 12, 1, l_sablony)
l_sablony = zm_vel('/home/ajuska/Plocha/bakalarka/oci/oko2.jpg', 5, 12, 1, l_sablony)
l_sablony = zm_vel('/home/ajuska/Plocha/bakalarka/oci/oko3.jpg', 5, 12, 1, l_sablony)
l_sablony = zm_vel('/home/ajuska/Plocha/bakalarka/oci/oko4.jpg', 5, 12, 1, l_sablony)
l_sablony = zm_vel('/home/ajuska/Plocha/bakalarka/oci/oko5.jpg', 5, 12, 1, l_sablony)
l_sablony = zm_vel('/home/ajuska/Plocha/bakalarka/oci/oko6.jpg', 5, 12, 1, l_sablony)
l_sablony = zm_vel('/home/ajuska/Plocha/bakalarka/oci/oko7.jpg', 5, 12, 1, l_sablony)
l_sablony = zm_vel('/home/ajuska/Plocha/bakalarka/oci/oko8.jpg', 5, 12, 1, l_sablony)
l_sablony = zm_vel('/home/ajuska/Plocha/bakalarka/oci/oko9.jpg', 5, 12, 1, l_sablony)
l_sablony = zm_vel('/home/ajuska/Plocha/bakalarka/oci/oko10.jpg', 5, 12, 1, l_sablony)

l_sablony = np.array(l_sablony)

sablonaL_vysl = []
for sablonaL in l_sablony:
    sablonaL_vysl.append(cv2.Canny(sablonaL, 10, 80))

oblicej_klas = cv2.CascadeClassifier('/home/ajuska/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_default.xml')

Cesta1 = []
label = []

for index in os.listdir("/home/ajuska/Plocha/bakalarka/test"):
    Cesta1.append(os.path.join("/home/ajuska/Plocha/bakalarka/test", index))

data1 = []
for index in Cesta1:
    nacteni_obr = cv2.imread(index)
    seda = cv2.cvtColor(nacteni_obr,cv2.COLOR_BGR2GRAY)
    seda = np.array(seda)
    data1.append(seda)

data1 = np.array(data1)
maximum = []
detekce = []
for index in data1:
    body = oblicej_klas.detectMultiScale(index)
    maximum = []
    for(x,y,w,h) in body:
        cv2.rectangle(index,(x,y),(x+300,y+400),(255,0,0),2)
        x,y = body [0][:2]
        orez = index[y: y + 400, x: x + 300]
        orez = cv2.GaussianBlur(orez,(5,5),0)
        pyplot.hist(orez.ravel(),256,[0,256])
        hrany = cv2.Canny(orez,50,150)

        velikost = hrany.shape
        sirka = velikost[1]
        vyska = velikost[0]

        for ind_sablona in range(0, len(sablonaL_vysl)):
            sablona = sablonaL_vysl[ind_sablona]
            velikost = sablona.shape
            korelace = cv2.matchTemplate(hrany, sablona, cv2.TM_CCORR_NORMED)
            min, max, min_poz, max_poz = cv2.minMaxLoc(korelace)
            maximum.append(max)
            #print(maximum)

        shoda = np.amax(maximum)
        if shoda >= 0.4:
           print (shoda)
           detekce.append(orez)


for i in range(0,len(detekce)):
    cv2.imshow('foto',detekce[i])
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

detekce = np.array(detekce)
detekce = detekce.reshape(detekce.shape[0],300,400,1)
detekce = detekce.astype('float32')
detekce /= 255


l_model = load_model('/home/ajuska/Plocha/bakalarka/model.h5')
out = l_model.predict_classes(detekce)
print(out)


