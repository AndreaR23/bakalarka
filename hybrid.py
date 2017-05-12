import numpy as np
import cv2
import os
from matplotlib import pyplot
from knihovna import zm_vel, loading
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D

# from picamera import PiCamera
# from time import sleep

# kamera = PiCamera()
# kamera.start_preview()
# sleep(3)
# kamera.capture('/home/pi/Desktop/foto.jpg')
# sleep(2)
# kamera.stop_preview()

# DETEKCE OBLICEJE

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

data1 = []
data1 = loading("/home/ajuska/Plocha/bakalarka/radek3",sablonaL_vysl, 1)
print(data1.shape)

oblicej_klas = cv2.CascadeClassifier('/home/ajuska/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_default.xml')

Cesta1 = []
label = []

for index in os.listdir("/home/ajuska/Plocha/bakalarka/radek3"):
    Cesta1.append(os.path.join("/home/ajuska/Plocha/bakalarka/radek3", index))

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

        velikost = hrany.shape
        sirka = velikost[1]
        vyska = velikost[0]

        # cv2.imshow('foto',orez)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        for ind_sablona in range(0, len(sablonaL_vysl)):
            sablona = sablonaL_vysl[ind_sablona]
            velikost = sablona.shape
            velR = velikost[0]
            velS = velikost[1]
            korelace = cv2.matchTemplate(hrany, sablona, cv2.TM_CCORR_NORMED)
            min, max, min_poz, max_poz = cv2.minMaxLoc(korelace)
            maximum.append(max)

        shoda = np.amax(maximum)
        if shoda >= 0.4:
           print (shoda)
           detekce.append(orez)
           label.append(1)

Cesta2 = []
for index in os.listdir("/home/ajuska/Plocha/bakalarka/pavel"):
    Cesta2.append(os.path.join("/home/ajuska/Plocha/bakalarka/pavel", index))

data2 = []
for index in Cesta2:
    nacteni_obr = cv2.imread(index)
    seda = cv2.cvtColor(nacteni_obr,cv2.COLOR_BGR2GRAY)
    seda = np.array(seda)
    data2.append(seda)

data2 = np.array(data2)
for index in data2:
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

        # cv2.imshow('foto',orez)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        for ind_sablona in range(0, len(sablonaL_vysl)):
            sablona = sablonaL_vysl[ind_sablona]
            velikost = sablona.shape
            velR = velikost[0]
            velS = velikost[1]
            korelace = cv2.matchTemplate(hrany, sablona, cv2.TM_CCORR_NORMED)
            min, max, min_poz, max_poz = cv2.minMaxLoc(korelace)
            maximum.append(max)

        shoda = np.amax(maximum)
        if shoda >= 0.4:
           print (shoda)
           detekce.append(orez)
           label.append(0)

detekce = np.array(detekce)
#print(detekce)

labely = np.array(label)
print(detekce.shape)
print(label)

# TRENOVACI A TESTOVACI DATABAZE
Xtren, Xtest, Ytren, Ytest = train_test_split(detekce, label, train_size=0.8)
print(Xtest.shape)

for i in range (0, len(Xtest)):
    cv2.imshow('image', Xtest[i])
    cv2.waitKey(2000)
    cv2.destroyAllWindows()

Xtren = np.array(Xtren)
Xtest = np.array(Xtest)
print(Xtren.shape)
Xtren = Xtren.reshape(Xtren.shape[0],300,400,1)
Xtest = Xtest.reshape(Xtest.shape[0],300,400,1)
Xtren = Xtren.astype('float32')
Xtest = Xtest.astype('float32')
Xtren /= 255
Xtest /= 255

Ytren = np.array(Ytren)
Ytest = np.array(Ytest)
tridy  = 2
Ytren = np_utils.to_categorical(Ytren, tridy)
Ytest = np_utils.to_categorical(Ytest, tridy)

print ('Rozmery trenovaci matice', Xtren.shape)
print ('Rozmery testovaci matice', Xtest.shape)
print('Rozmery Ytren', Ytren.shape)
print('Rozmery Ytest', Ytest.shape)

# TVORBA MODELU
model = Sequential()

model.add(Conv2D(12, 5, 5, activation = 'relu', input_shape=(300,400,1), init='he_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(25, 5, 5, activation = 'relu', init='he_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(180, activation = 'relu', init='he_normal'))
model.add(Dropout(0.5))
model.add(Dense(100, activation = 'relu', init='he_normal'))
model.add(Dropout(0.5))
model.add(Dense(tridy, activation = 'softmax', init='he_normal'))

model.summary()

# KOMPILACE A PRUBEH
model.compile(loss='binary_crossentropy', optimizer="SGD", metrics=['accuracy'])
model.fit(Xtren, Ytren, batch_size=64, nb_epoch=50, verbose=1, validation_data=(Xtest, Ytest))

# ODHAD
loss, accuracy = model.evaluate(Xtest,Ytest, verbose=0)

model.save('/home/ajuska/Plocha/bakalarka/model.h5')

print('Loss',loss)
print('Accurancy',accuracy)
pred_tridy = model.predict_classes(Xtest)
print(pred_tridy)








