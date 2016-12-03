# Author: Jose Miguel Buenaposada 2015.
# Simple MSER application for traffic sign window proposal generation
import cv2
import numpy as np
# import matplotlib as plt
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda

mser = cv2.MSER_create()

test_dir = 'Imagenes Deteccion/test/'
TRAIN_DIR = 'Imagenes Deteccion/train/'
test_ext = ('.jpg', '.ppm')

windowArray = []
windowArray10 = []
labels = []
clasificadorLower = lda()
clasificadorHigher = lda()

def test():
        for filename in os.listdir(test_dir):
            if os.path.splitext(filename)[1].lower() in test_ext:
                print ("Test, processing ", filename, "\n");
                full_path = os.path.join(test_dir, filename)
                I = cv2.imread(full_path)
                Icopy = I.copy()
                Igray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

                regions = mser.detectRegions(Igray, None)
                rects = [cv2.boundingRect(p.reshape(-1,1,2)) for p in regions]
                for r in rects:
                   x,y,w,h = r
                   # Simple aspect ratio filtering
                   aratio = float(w)/float(h)
                   if (aratio > 1.2) or (aratio < 0.8):
                       continue
                   cv2.rectangle(Icopy, (x, y), (x + w, y + h), (0, 255, 0), 2)

                cv2.imshow('img', Icopy)
                cv2.waitKey(0)
def LDA():
    print("·")
    lowerWindow = np.vstack(windowArray)
    upperWindow = np.vstack(windowArray10)
    E = np.array(labels)
    clasificadorLower.fit(lowerWindow,E)
    clasificadorHigher.fit(upperWindow,E)
    


def train():
    global windowArray
    global labels
    for folder in os.listdir(TRAIN_DIR):
        parcial_path = os.path.join(TRAIN_DIR, folder)
        if(parcial_path.__contains__("Otros")):
            continue
        for subfolder in os.listdir(parcial_path):
            subF = os.path.join(parcial_path, subfolder)
            for filename in os.listdir(subF):
                if os.path.splitext(filename)[1].lower() in test_ext:
                    lista = []
                    print("Test, processing ", subF+"\\"+filename, "\n")
                    full_path = os.path.join(subF, filename)
                    I = cv2.imread(full_path)
                    Icopy = I.copy()
                    Igray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
                    shape = Icopy.shape
                    regions = mser.detectRegions(Igray, None)
                    rects = [cv2.boundingRect(p.reshape(-1, 1, 2)) for p in regions]
                    for r in rects:
                        x, y, w, h = r
                        # Simple aspect ratio filtering
                        aratio = float(w) / float(h)
                        if (aratio > 1.2) or (aratio < 0.8):
                            continue
                        # cv2.rectangle(Icopy, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        lista.append([x,y,x+w,y+h])
                    miLista= set(tuple(i) for i in lista)
                    if(lista.__len__()>0):
                        max = lista[0]
                       # for index in range (1,miLista.__len__()):
                        for elem in miLista:
                            if max[0]>elem[0] and max[3]<elem[3]:
                                max = elem
                        area1 = max
                        #### LLAMADA A PROCESO LDA
                        area2 = [int((max[0]*0.1)+(max[0])),int((max[1]*0.1)+(max[1])),int((max[2]*0.1)+(max[2])),int((max[3]*0.1)+(max[3]))]
                        #### LLAMADA A PROCESO LDA
                        window = Icopy[max[1]:max[1] + max[3], max[0]:max[0] + max[2]]
                        # windowArray.append(window)
                        window = cv2.resize(window, (10, 10), None, 0, 0, cv2.INTER_NEAREST)
                        lab = None
                        if subF.__contains__("Peligro"):
                            lab = "Peligro"
                        elif subF.__contains__("Prohibicion"):
                            lab = "Prohibicion"
                        else:
                            lab = "Otros"
                        labels.append(lab)
                        windowArray.append(window.ravel())
                        # cv2.imshow('img', window)
                        # cv2.imshow('img2', windowEXT)
                        # cv2.waitKey()
                        # cv2.destroyAllWindows()
                        if area2[0]<0 or area2[1]<0 or area2[2]>shape[0] or area2[3]>shape[1]:
                            windowArray10.append(window.ravel())
                        else:
                            windowEXT = Icopy[area2[1]:area2[1] + area2[3], area2[0]:area2[0] + area2[2]]
                            windowEXT = cv2.resize(windowEXT, (10, 10), None, 0, 0, cv2.INTER_NEAREST)
                            windowArray10.append(windowEXT.ravel())
                        # windowArray.append(windowEXT)
                        # cv2.rectangle(Icopy, (max[0], max[1]), (max[2], max[3]), (0, 255, 0), 2)
                        # cv2.imshow('img', Icopy)
                        # cv2.waitKey(0)
                        # cv2.destroyWindow("img")



train()
print(windowArray)