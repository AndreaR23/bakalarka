import numpy as np
import cv2
from matplotlib import pyplot 
from library import change_size

image = cv2.imread('/home/ajuska/Plocha/bakalarka/aja/aja0009.jpg')
grey_colors = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
grey_colors2 = cv2.GaussianBlur(grey_colors, (5, 5), 0)
pyplot.hist(grey_colors2.ravel(), 256, [0, 256])
edges = cv2.Canny(grey_colors2, 30, 60)
cv2.imshow('foto', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# sablony
eye_template = []
eye_template = change_size('/home/ajuska/Plocha/bakalarka/oci/oko1.jpg', 3, 15, 1, eye_template)
eye_template = change_size('/home/ajuska/Plocha/bakalarka/oci/oko2.jpg', 3, 15, 1, eye_template)
eye_template = change_size('/home/ajuska/Plocha/bakalarka/oci/oko3.jpg', 3, 15, 1, eye_template)
eye_template = change_size('/home/ajuska/Plocha/bakalarka/oci/oko4.jpg', 3, 15, 1, eye_template)
eye_template = change_size('/home/ajuska/Plocha/bakalarka/oci/oko5.jpg', 3, 15, 1, eye_template)
eye_template = change_size('/home/ajuska/Plocha/bakalarka/oci/oko6.jpg', 3, 15, 1, eye_template)
eye_template = change_size('/home/ajuska/Plocha/bakalarka/oci/oko7.jpg', 3, 15, 1, eye_template)
eye_template = change_size('/home/ajuska/Plocha/bakalarka/oci/oko8.jpg', 3, 15, 1, eye_template)
eye_template = change_size('/home/ajuska/Plocha/bakalarka/oci/oko9.jpg', 3, 15, 1, eye_template)
eye_template = change_size('/home/ajuska/Plocha/bakalarka/oci/oko10.jpg', 3, 15, 1, eye_template)

eye_template = np.array(eye_template)

template_fin = []
for templ in eye_template:
    template_fin.append(cv2.Canny(templ, 10, 80))

dimensions = edges.shape
width = dimensions[1]
high = dimensions[0]
print(dimensions)

for s in range(1, width):
    for r in range (0, high):
        if edges[r, s] == 0:
           edges[r, s] = 5

# Vyber sablony
max = 0
coordinates = []
loc_temp = []
PERCENT = 0.15
STEP = 4

for column in range (0, width - 120, STEP): #Prochazeni sloupcu a radku v puvodnim obraze, bez okrajovych pixelu
    for row in range (0, high - 100, STEP):
        for ind_templ in range(0, len(template_fin)):
            template = template_fin[ind_templ]
            dimensions = template.shape
            dimR = dimensions[0]
            dimC = dimensions[1]
            area = edges[row:dimR + row, column:dimC + column]
            if ((np.sum(area)*100))/255 / (dimR*dimC) > 8:
                compare = np.isclose(area, template)
                compare = (np.sum(compare) * 100) / (np.sum(template) / 255)
                if max < compare:
                   max = compare
                   coordinates = [column, row]
                   loc_temp = ind_templ

x = coordinates[0]
y = coordinates[1]
y2 = coordinates[1] + 50
x2 = coordinates[0] + 50
cv2.rectangle(image, (x, y), (x2, y2), (255, 0, 0), 2)
cv2.imshow('foto', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
