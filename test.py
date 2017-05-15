from keras.models import load_model
from library import change_size, loading
import numpy as np
import cv2
from keras.models import model_from_json

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

PATH_XML = '/home/ajuska/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_default.xml'
crop = []
labels = []

loading("/home/ajuska/Plocha/bakalarka/test", templ_final, 0, PATH_XML, crop, labels)

crop = np.array(crop)

for i in range(0,len(crop)):
    cv2.imshow('foto',crop[i])
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

test = np.array(crop)
test = test.reshape(test.shape[0],300,400,1)
test = test.astype('float32')
test /= 255

# l_model = load_model('/home/ajuska/Plocha/bakalarka/model.h5')
# output = l_model.predict_classes(test)
# print(output)

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")

loaded_model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
score = loaded_model.predict_classes(test, verbose=0)
print(score)

