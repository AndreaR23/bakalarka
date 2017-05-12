import cv2
import numpy as np
from matplotlib import pyplot
import os

def zm_vel (sab, nejmensi, nejvetsi, krok, pole):
    zena1 = cv2.imread(sab)
    s_zena1 = cv2.cvtColor(zena1, cv2.COLOR_BGR2GRAY)
    hrany_z1 = cv2.GaussianBlur(s_zena1,(3,3),0)
    for i in range(nejmensi,nejvetsi,krok):
        j = i/10.0
        pole.append(cv2.resize(hrany_z1,None, fx=j, fy=j, interpolation = cv2.INTER_CUBIC))
    return pole

def loading (cesta, template, label):

    Cesta1 = []
    trida = []
    oblicej_klas = cv2.CascadeClassifier('/home/ajuska/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_default.xml')

    for index in os.listdir(cesta):
        Cesta1.append(os.path.join(cesta, index))

    data1 = []
    for index in Cesta1:
        nacteni_obr = cv2.imread(index)
        seda = cv2.cvtColor(nacteni_obr,cv2.COLOR_BGR2GRAY)
        seda = np.array(seda)
        data1.append(seda)

    data1 = np.array(data1)
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

            for ind_sablona in range(0, len(template)):
                sablona = template[ind_sablona]
                korelace = cv2.matchTemplate(hrany, sablona, cv2.TM_CCORR_NORMED)
                min, max, min_poz, max_poz = cv2.minMaxLoc(korelace)
                maximum.append(max)
                #print(maximum)

            shoda = np.amax(maximum)
            if shoda >= 0.4:
               detekce.append(orez)
               trida.append(label)
    return data1
