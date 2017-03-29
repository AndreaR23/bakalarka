# IMPORT KNIHOVEN
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split

# NACTENI DAT
DatasetCesta = []
for index in os.listdir("/home/ajuska/Plocha/bakalarka/emma"):
    DatasetCesta.append(os.path.join("/home/ajuska/Plocha/bakalarka/emma", index))

data = []
popisy = []

for index in DatasetCesta:
    nacteni_obr = cv2.imread(index)
    seda = cv2.cvtColor(nacteni_obr,cv2.COLOR_BGR2GRAY)
    data.append(seda)
    nacteni_pop = int(os.path.split(index)[1].split(".")[0].replace("subject", " ")) - 1
    popisy.append(nacteni_pop)

popisy = np.array(popisy)
data = np.array(data)
# vel1 = popisy.shape
# vel2 = data.shape
# print(vel1)
# print(vel2)

# DETEKCE OBLICEJE
oblicej_klas = cv2.CascadeClassifier('/home/ajuska/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_default.xml')

detekce = []
for index in data:
    body = oblicej_klas.detectMultiScale(index)
    for(x,y,w,h) in body:
        cv2.rectangle(index,(x,y),(x+w,y+h),(255,0,0),2)
        x,y = body [0][:2]
        orez = index[y: y + h, x: x + w]
    detekce.append(orez)
        # cv2.imshow('foto',orez)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

detekce = np.array(detekce)
# vel = detekce.shape
# print (vel)

# TRENOVACI A TESTOVACI DATABAZE
Xtren, Xtest, Ytren, Ytest = train_test_split(detekce, popisy, train_size=0.9, random_state=5)
Xtren = np.array(Xtren, dtype=object)
Xtest = np.array(Xtest, dtype =object)
Xtren = Xtren.reshape(9,1)
Xtest = Xtest.reshape(1,1)
Xtren /= 255
Xtest /= 255

Ytren = np.array(Ytren)
Ytest = np.array(Ytest)
tridy  = 15
Ytren = np_utils.to_categorical(Ytren, tridy)
Ytest = np_utils.to_categorical(Ytest, tridy)

print ('Rozměry trénovací matice', Xtren.shape)
print ('Rozmery testovací matice', Xtest.shape)

# TVORBA MODELU
model = Sequential()
model.add(Dense(512,input_shape=(Xtren.shape[1],)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(tridy))
model.add(Activation('softmax'))

model.summary()

# KOMPILACE A PRUBEH
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
model.fit(Xtren, Ytren, batch_size=64, nb_epoch=50, verbose=1, validation_data=(Xtest, Ytest))
