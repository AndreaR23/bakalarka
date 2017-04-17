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
labely = []
for index in os.listdir("/home/ajuska/Plocha/bakalarka/true"):
    DatasetCesta.append(os.path.join("/home/ajuska/Plocha/bakalarka/true", index))
    labely.append(1)
for index in os.listdir("/home/ajuska/Plocha/bakalarka/false"):
    DatasetCesta.append(os.path.join("/home/ajuska/Plocha/bakalarka/false", index))
    labely.append(0)

data = []
for index in DatasetCesta:
    nacteni_obr = cv2.imread(index)
    seda = cv2.cvtColor(nacteni_obr,cv2.COLOR_BGR2GRAY)
    seda = np.array(seda)
    data.append(seda)

labely = np.array(labely)
data = np.array(data)

# DETEKCE OBLICEJE
oblicej_klas = cv2.CascadeClassifier('/home/ajuska/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_default.xml')

detekce = []
for index in data:
    body = oblicej_klas.detectMultiScale(index)
    for(x,y,w,h) in body:
        cv2.rectangle(index,(x,y),(x+200,y+200),(255,0,0),2)
        x,y = body [0][:2]
        orez = index[y: y + 200, x: x + 200]
        # cv2.imshow('foto',orez)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    detekce.append(orez)

detekce = np.array(detekce)

# TRENOVACI A TESTOVACI DATABAZE
Xtren, Xtest, Ytren, Ytest = train_test_split(detekce, labely, train_size=0.8)

# cv2.imshow('foto',Xtest[1])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

Xtren = np.array(Xtren)
Xtest = np.array(Xtest)
print(Xtren.shape)
Xtren = Xtren.reshape(14,200*200)
Xtest = Xtest.reshape(4,200*200)
Xtren = Xtren.astype('float32')
Xtest = Xtest.astype('float32')
Xtren /= 255
Xtest /= 255

Ytren = np.array(Ytren)
Ytest = np.array(Ytest)
tridy  = 2
Ytren = np_utils.to_categorical(Ytren, tridy)
Ytest = np_utils.to_categorical(Ytest, tridy)

print ('Rozměry trénovací matice', Xtren.shape)
print ('Rozmery testovací matice', Xtest.shape)
print('Rozmery Ytren', Ytren.shape)
print('Rozmery Ytest', Ytest.shape)

# TVORBA MODELU
model = Sequential()
model.add(Dense(512,input_shape=(Xtren.shape[1],), init='uniform', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512,init='uniform', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(tridy, init = 'uniform', activation='softmax'))

model.summary()

# KOMPILACE A PRUBEH
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
model.fit(Xtren, Ytren, batch_size=64, nb_epoch=10, verbose=1, validation_data=(Xtest, Ytest))

# ODHAD
loss, accuracy = model.evaluate(Xtest,Ytest, verbose=0)

print('Loss',loss)
print('Accurancy',accuracy)
pred_tridy = model.predict_classes(Xtest)
print(pred_tridy)

