import numpy as np
import cv2
from library import change_size, loading, loading2, network
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split

CLASSES = 12
PATH_XML = '/home/ajuska/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_default.xml'

# LOAD AND PREPROCESSING EYE TEMPLATES
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

# CALL FUNCTION FOR LOADING CROPS WITH DETECTED FACES AND MAKING TRAINING DATABASE
crop = []
labels = []
loading("/home/ajuska/Plocha/bakalarka/radek3", templ_final, 0, PATH_XML, crop, labels)
loading("/home/ajuska/Plocha/bakalarka/pavel", templ_final, 1, PATH_XML, crop, labels)
loading("/home/ajuska/Plocha/bakalarka/anicka", templ_final, 2, PATH_XML, crop, labels)
loading("/home/ajuska/Plocha/bakalarka/david", templ_final, 3, PATH_XML, crop, labels)
loading("/home/ajuska/Plocha/bakalarka/zbynda", templ_final, 4, PATH_XML, crop, labels)
loading("/home/ajuska/Plocha/bakalarka/aja", templ_final, 5, PATH_XML, crop, labels)
print(labels)
loading2("/home/ajuska/Plocha/bakalarka/jura", 6, crop, labels)
loading2("/home/ajuska/Plocha/bakalarka/jiri", 7, crop, labels)
loading2("/home/ajuska/Plocha/bakalarka/matejk", 8, crop, labels)
loading2("/home/ajuska/Plocha/bakalarka/jarda", 9, crop, labels)
loading2("/home/ajuska/Plocha/bakalarka/zdenka", 10, crop, labels)
loading2("/home/ajuska/Plocha/bakalarka/zikina", 11, crop, labels)
print(labels)

crop = np.array(crop)
label = np.array(labels)

print(crop.shape)

# SPLIT TO TRAINING AND TESTING DATABASE
Xtrain, Xtest, Ytrain, Ytest = train_test_split(crop, label, train_size=0.9)
print(Xtest.shape)

# DISPLAY TESTING IMAGES
for i in range (0, len(Xtest)):
    print(Xtest[i].shape)
    cv2.imshow('image', Xtest[i])
    cv2.waitKey(3000)
    cv2.destroyAllWindows()

# PROCESS TESTING IMAGES TO RIGHT FORMAT
Xtrain = np.array(Xtrain)
Xtest = np.array(Xtest)
print(Xtrain.shape)
Xtrain = Xtrain.reshape(Xtrain.shape[0], 400, 400, 1)
Xtest = Xtest.reshape(Xtest.shape[0], 400, 400, 1)
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

# PREDICT CLASSES OF TESTING IMAGES
loss, acurancy, pred_classes = network((400, 400, 1), Xtrain, Ytrain, Xtest, Ytest, CLASSES)
print(pred_classes)

# ASSIGN A NAME TO THE CLASS
score = np.array(pred_classes)
for index in range(0,len(score)):
    if score[index] == 0:
        print ('Radek')
    elif score[index] == 1:
        print('Pavel')
    elif score[index] == 2:
        print('Anicka')
    elif score[index] == 3:
        print('David')
    elif score[index] == 4:
        print('Zbynda')
    elif score[index] == 5:
        print('Aja')
    elif score[index] == 6:
        print('Jura')
    elif score[index] == 7:
        print('Jiri')
    elif score[index] == 8:
        print('Matej')
    elif score[index] == 9:
        print('Jarda')
    elif score[index] == 10:
        print('Zdenka')
    elif score[index] == 11:
        print('Petr')







