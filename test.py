from library import change_size, loading, loading2
import numpy as np
import cv2
from keras.models import model_from_json

# LOAD AND PREPROCCESS EYE TEMPLATES
eye_template = []
eye_template = change_size('/home/Pi/Plocha/bakalarka/oci/oko1.jpg', 5, 12, 1, eye_template)
eye_template = change_size('/home/Pi/Plocha/bakalarka/oci/oko2.jpg', 5, 12, 1, eye_template)
eye_template = change_size('/home/Pi/Plocha/bakalarka/oci/oko3.jpg', 5, 12, 1, eye_template)
eye_template = change_size('/home/Pi/Plocha/bakalarka/oci/oko4.jpg', 5, 12, 1, eye_template)
eye_template = change_size('/home/Pi/Plocha/bakalarka/oci/oko5.jpg', 5, 12, 1, eye_template)
eye_template = change_size('/home/Pi/Plocha/bakalarka/oci/oko6.jpg', 5, 12, 1, eye_template)
eye_template = change_size('/home/Pi/Plocha/bakalarka/oci/oko7.jpg', 5, 12, 1, eye_template)
eye_template = change_size('/home/Pi/Plocha/bakalarka/oci/oko8.jpg', 5, 12, 1, eye_template)
eye_template = change_size('/home/Pi/Plocha/bakalarka/oci/oko9.jpg', 5, 12, 1, eye_template)
eye_template = change_size('/home/Pi/Plocha/bakalarka/oci/oko10.jpg', 5, 12, 1, eye_template)
eye_template = np.array(eye_template)

templ_final = []
for index in eye_template:
    templ_final.append(cv2.Canny(index, 10, 80))

PATH_XML = '/home/Pi/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_default.xml'
crop = []
labels = []

# CALL FUNCTION FOR LOADING CROPS WITH FACES
loading("/home/Pi/Plocha/bakalarka/test11", templ_final, 0, PATH_XML, crop, labels)
# loading2("/home/ajuska/Plocha/bakalarka/test2", 0, crop, labels)

crop = np.array(crop)

# DISPLAY TESTING IMAGES
for i in range(0,len(crop)):
    cv2.imshow('foto',crop[i])
    cv2.waitKey(3000)
    cv2.destroyAllWindows()

# PROCESS TESTING IMAGES TO RIGHT FORMAT
test = np.array(crop)
test = test.reshape(test.shape[0],400,400,1)
test = test.astype('float32')
test /= 255

# LOAD PRE-TRAINED MODEL
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")

# COMPILE MODELAND PREDICT CLASSES OF TESTING IMAGES
loaded_model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
score = loaded_model.predict_classes(test, verbose=0)
print(score)

# ASSIGN A NAME TO THE CLASS
score = np.array(score)
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
