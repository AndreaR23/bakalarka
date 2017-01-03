import numpy as np
import cv2
from matplotlib import pyplot 
from picamera import PiCamera
from time import sleep
from knihovna import zm_vel

kamera = PiCamera()
kamera.resolution=(640,480)
kamera.start_preview()
sleep(10)
kamera.capture('/home/pi/Desktop/foto.jpg')
sleep(2)
kamera.stop_preview()

fotka = cv2.imread('/home/pi/Desktop/foto.jpg')
sedoton = cv2.cvtColor(fotka,cv2.COLOR_BGR2GRAY)
sedoton2 = cv2.GaussianBlur(sedoton,(5,5),0)
pyplot.hist(sedoton2.ravel(),256,[0,256])
hrany = cv2.Canny(sedoton2,50,150)

# sablony
l_sablony = []
l_sablony = zm_vel('/home/pi/Desktop/oko1.jpg', 5, 12, 1, l_sablony)
l_sablony = zm_vel('/home/pi/Desktop/oko2.jpg', 5, 12, 1, l_sablony)
l_sablony = zm_vel('/home/pi/Desktop/oko3.jpg', 5, 12, 1, l_sablony)
l_sablony = zm_vel('/home/pi/Desktop/oko4.jpg', 5, 12, 1, l_sablony)
l_sablony = zm_vel('/home/pi/Desktop/oko5.jpg', 5, 12, 1, l_sablony)
l_sablony = zm_vel('/home/pi/Desktop/oko6.jpg', 5, 12, 1, l_sablony)
l_sablony = zm_vel('/home/pi/Desktop/oko7.jpg', 5, 12, 1, l_sablony)
l_sablony = zm_vel('/home/pi/Desktop/oko8.jpg', 5, 12, 1, l_sablony)
l_sablony = zm_vel('/home/pi/Desktop/oko9.jpg', 5, 12, 1, l_sablony)
l_sablony = zm_vel('/home/pi/Desktop/oko10.jpg', 5, 12, 1, l_sablony)

l_sablony = np.array(l_sablony)

sablonaL_vysl = []
for sablonaL in l_sablony:
    sablonaL_vysl.append(cv2.Canny(sablonaL, 10, 80))

velikost = hrany.shape
sirka = velikost[1]
vyska = velikost[0]
print(velikost)

for s in range(1,sirka):
    for r in range (0,vyska):
        if hrany[r, s] == 0:
           hrany[r, s] = 5

# Vyber sablony
max = 0
souradnice = []
poz_sablona = []
PROCENTA = 0.15
KROK = 4

for sloupec in range (0, sirka - 120, KROK): #Prochazeni sloupcu a radku v puvodnim obraze, bez okrajovych pixelu
    for radek in range (0, vyska - 100, KROK):
        for ind_sablona in range(0, len(sablonaL_vysl)):
            sablona = sablonaL_vysl[ind_sablona]
            velikost = sablona.shape
            velR = velikost[0]
            velS = velikost[1]
            oblast = hrany[radek:velR+radek, sloupec:velS+sloupec]
            if ((np.sum(oblast)*100))/255 / (velR*velS) > 8:
                porovnani = np.isclose(oblast, sablona)
                porovnani = (np.sum(porovnani) * 100) / (np.sum(sablona) / 255)
                if max < porovnani:
                   max = porovnani
                   souradnice = [sloupec, radek]
                   poz_sablona = ind_sablona

x = souradnice[0] 
y = souradnice[1] 
y2 = souradnice[1] + 50
x2 = souradnice[0] + 50
cv2.rectangle(fotka,(x,y),(x2,y2),(255,0,0),2)
cv2.imshow('foto',fotka)
cv2.waitKey(0)
cv2.destroyAllWindows()


