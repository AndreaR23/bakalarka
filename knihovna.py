import cv2

def zm_vel (sab, nejmensi, nejvetsi, krok, pole):
    zena1 = cv2.imread(sab)
    s_zena1 = cv2.cvtColor(zena1, cv2.COLOR_BGR2GRAY)
    hrany_z1 = cv2.GaussianBlur(s_zena1,(3,3),0)
    for i in range(nejmensi,nejvetsi,krok):
        j = i/10.0
        pole.append(cv2.resize(hrany_z1,None, fx=j, fy=j, interpolation = cv2.INTER_CUBIC))
    return pole
