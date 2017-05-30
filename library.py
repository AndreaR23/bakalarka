import cv2
import numpy as np
from matplotlib import pyplot
import os
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D

# FUNCTION FOR EYE-TEMPLATES PREPROCESSING
def change_size (temp, smallest, biggest, step, array):
    woman = cv2.imread(temp)
    temp_woman = cv2.cvtColor(woman, cv2.COLOR_BGR2GRAY)
    temp_edges = cv2.GaussianBlur(temp_woman,(3,3),0)
    for i in range(smallest, biggest, step):
        j = i/10.0
        array.append(cv2.resize(temp_edges, None, fx=j, fy=j, interpolation = cv2.INTER_CUBIC))
    return array

# FUNCTION FOR FACES DETECTION IN IMAGES FROM LOADING FILES
def loading (path, template, label, path_xml, crop, classes):

    Path1 = []
    face_class = cv2.CascadeClassifier(path_xml)

    for index in os.listdir(path):
        Path1.append(os.path.join(path, index))

    data1 = []
    for index in Path1:
        load_img = cv2.imread(index)
        grey = cv2.cvtColor(load_img,cv2.COLOR_BGR2GRAY)
        grey = np.array(grey)
        data1.append(grey)

    data1 = np.array(data1)
    for index in data1:
        points = face_class.detectMultiScale(index)
        maximum = []
        for(x,y,w,h) in points:
            cv2.rectangle(index,(x,y),(x+400,y+400),(255,0,0),2)
            x,y = points [0][:2]
            detection = index[y: y + 400, x: x + 400]
            detection = np.array(detection)
            # print(detection.shape[0])

            if (detection.shape[0] ==400) & (detection.shape[1] ==400):
               detection = cv2.GaussianBlur(detection,(5,5),0)
               pyplot.hist(detection.ravel(),256,[0,256])
               edges = cv2.Canny(detection,50,150)

               for ind_temp in range(0, len(template)):
                   templ = template[ind_temp]
                   correlation = cv2.matchTemplate(edges, templ, cv2.TM_CCORR_NORMED)
                   min, max, min_loc, max_loc = cv2.minMaxLoc(correlation)
                   maximum.append(max)
                   #print(maximum)

               same = np.amax(maximum)
               if same >= 0.05:
                  crop.append(detection)
                  classes.append(label)
               break

# FUNCTION FOR LOADING PRE-PREPARED CROPS OF DETECTED FACES FROM ANOTHER FILE
def loading2 (path, label, crop, classes):

    Path1 = []

    for index in os.listdir(path):
        Path1.append(os.path.join(path, index))

    for index in Path1:
        load_img = cv2.imread(index)
        r = 400 / load_img.shape[1]
        dim = (400, int(load_img.shape[0] * r))
        resized = cv2.resize(load_img, dim, interpolation = cv2.INTER_AREA)
        grey = cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
        grey = np.array(grey)

        if (grey.shape[0] ==400) & (grey.shape[1] ==400):
            crop.append(grey)
            classes.append(label)

# FUNCTION FOR DEFINED CONVOLUTIONAL NEURAL NETWORK
def network (dimensions, Xtrain, Ytrain, Xtest, Ytest, classes_num):
    model = Sequential()

    model.add(Conv2D(12, 5, 5, activation = 'relu', input_shape=(dimensions), init='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(25, 5, 5, activation = 'relu', init='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(180, activation = 'relu', init='he_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation = 'relu', init='he_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(classes_num, activation ='softmax', init='he_normal'))

    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer="SGD", metrics=['accuracy'])
    model.fit(Xtrain, Ytrain, batch_size=64, nb_epoch=50, verbose=1, validation_data=(Xtest, Ytest))

    loss, accuracy = model.evaluate(Xtest, Ytest, verbose=0)
    pred_classes = model.predict_classes(Xtest)

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")

    return loss, accuracy, pred_classes




