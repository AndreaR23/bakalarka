import numpy as np
import cv2
from library import change_size, loading, network
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split

# from picamera import PiCamera
# from time import sleep
#
# camera = PiCamera()
# camera.start_preview()
# sleep(3)
# camera.capture('/home/pi/Desktop/foto.jpg')
# sleep(2)
# camera.stop_preview()

CLASSES = 5
PATH_XML = '/home/ajuska/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_default.xml'

eye_template = []
eye_template = change_size('/home/ajuska/Plocha/bakalarka/oci/oko1.jpg', 5, 12, 1, eye_template)
eye_template = change_size('/home/ajuska/Plocha/bakalarka/oci/oko2.jpg', 5, 12, 1, eye_template)
eye_template = change_size('/home/ajuska/Plocha/bakalarka/oci/oko3.jpg', 5, 12, 1, eye_template)
eye_template = change_size('/home/ajuska/Plocha/bakalarka/oci/oko4.jpg', 5, 12, 1, eye_template)
eye_template = change_size('/home/ajuska/Plocha/bakalarka/oci/oko5.jpg', 5, 12, 1, eye_template)
eye_template = change_size('/home/ajuska/Plocha/bakalarka/oci/oko6.jpg', 5, 12, 1, eye_template)
eye_template = change_size('/home/ajuska/Plocha/bakalarka/oci/oko7.jpg', 5, 12, 1, eye_template)
eye_template = change_size('/home/ajuska/Plocha/bakalarka/oci/oko8.jpg', 5, 12, 1, eye_template)
eye_template = change_size('/home/ajuska/Plocha/bakalarka/oci/oko9.jpg', 5, 12, 1, eye_template)
eye_template = change_size('/home/ajuska/Plocha/bakalarka/oci/oko10.jpg', 5, 12, 1, eye_template)

eye_template = np.array(eye_template)

templ_final = []
for index in eye_template:
    templ_final.append(cv2.Canny(index, 10, 80))

crop = []
labels = []
loading("/home/ajuska/Plocha/bakalarka/radek3", templ_final, 0, PATH_XML, crop, labels)
loading("/home/ajuska/Plocha/bakalarka/pavel", templ_final, 1, PATH_XML, crop, labels)
loading("/home/ajuska/Plocha/bakalarka/anicka", templ_final, 2, PATH_XML, crop, labels)
loading("/home/ajuska/Plocha/bakalarka/david", templ_final, 3, PATH_XML, crop, labels)
loading("/home/ajuska/Plocha/bakalarka/zbynda", templ_final, 4, PATH_XML, crop, labels)

crop = np.array(crop)
label = np.array(labels)

print(crop.shape)
print(label)

# TRENOVACI A TESTOVACI DATABAZE
Xtrain, Xtest, Ytrain, Ytest = train_test_split(crop, label, train_size=0.9)
print(Xtest.shape)

for i in range (0, len(Xtest)):
    print(Xtest[i].shape)
    cv2.imshow('image', Xtest[i])
    cv2.waitKey(2000)
    cv2.destroyAllWindows()

Xtrain = np.array(Xtrain)
Xtest = np.array(Xtest)
print(Xtrain.shape)
Xtrain = Xtrain.reshape(Xtrain.shape[0], 300, 400, 1)
Xtest = Xtest.reshape(Xtest.shape[0], 300, 400, 1)
Xtrain = Xtrain.astype('float32')
Xtest = Xtest.astype('float32')
Xtrain /= 255
Xtest /= 255

Ytrain = np.array(Ytrain)
Ytest = np.array(Ytest)
Ytrain = np_utils.to_categorical(Ytrain, CLASSES)
Ytest = np_utils.to_categorical(Ytest, CLASSES)

print('Rozmery trenovaci matice', Xtrain.shape)
print('Rozmery testovaci matice', Xtest.shape)
print('Rozmery Ytren', Ytrain.shape)
print('Rozmery Ytest', Ytest.shape)

loss, acurancy, pred_classes = network((300, 400, 1), Xtrain, Ytrain, Xtest, Ytest, CLASSES)








