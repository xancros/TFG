# Author: Jose Miguel Buenaposada 2015.
# Simple MSER application for traffic sign window proposal generation
import cv2
import numpy as np
# import matplotlib as plt
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
np.set_printoptions(suppress=True)
mser = cv2.MSER_create()

test_dir = 'Imagenes Deteccion/test/'
TRAIN_DIR = 'Imagenes Deteccion/train/'
test_ext = ('.jpg', '.ppm')
diccionario = {'Prohibicion':0,'Peligro':1,'Stop':2,'Otros':3}
windowArray = []
windowArray10 = []
labels = []
clasificadorLower = lda()
clasificadorHigher = lda()
res = []

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
    global windowArray
    global windowArray10
    global labels
    print("·")
    lowerWindow = np.vstack(windowArray)
    upperWindow = np.vstack(windowArray10)
    E = np.array(labels)
    clasificadorLower.fit(lowerWindow,E)
    vectorCLower=clasificadorLower.transform(lowerWindow)
    vectorCUpper=clasificadorHigher.fit_transform(upperWindow,E)
    CRU = vectorCUpper.astype(np.float32,copy=True)
    CLU = cv2.ml.NormalBayesClassifier_create()
    CLU.train(CRU,cv2.ml.ROW_SAMPLE,E)
    CRL = vectorCLower.astype(np.float32, copy=True)
    CLL = cv2.ml.NormalBayesClassifier_create()
    CLL.train(CRL, cv2.ml.ROW_SAMPLE, E)
    print()
    ###########################

    test = cv2.imread(TRAIN_DIR+"/Otros/06/00000.ppm")
    Icopy = test.copy()
    Igray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    shape = Icopy.shape
    regions = mser.detectRegions(Igray, None)
    rects = [cv2.boundingRect(p.reshape(-1, 1, 2)) for p in regions]
    file = open("salidaRe.txt",'w')
    for r in rects:
        x, y, w, h = r
        # Simple aspect ratio filtering
        aratio = float(w) / float(h)
        if (aratio > 1.2) or (aratio < 0.8):
            continue
        areaTest = Icopy[y:y + h, x:x + w]
        # cv2.imshow("imagen", test)
        # cv2.imshow("area",areaTest)
        # cv2.waitKey(1200)
        cv2.destroyAllWindows()
        senal = cv2.resize(areaTest, (25, 25), None, 0, 0, cv2.INTER_NEAREST)
        signal = (np.asarray(senal)).reshape(1,-1)
        signalT = clasificadorLower.transform(signal)
        vectorSignalT = signalT.astype(np.float32)
        _, Yhat1, prob1 = CLL.predictProb(vectorSignalT)
        file.write("Clase supuesta -> " +str((Yhat1[0])[0])+"\n")
        print("Clase supuesta -> ",(Yhat1[0])[0])
    ###########################
    file.close()

def train():
    global windowArray
    global windowArray10
    global labels
    global res
    count = 0
    count2 = 0
    listaImagenes = []
    numOtros, numStop,numPeligro,numPro = 0,0,0,0
    for folder in os.listdir(TRAIN_DIR):
        parcial_path = os.path.join(TRAIN_DIR, folder)
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
                        window = cv2.resize(window, (25, 25), None, 0, 0, cv2.INTER_NEAREST)
                        lab = None
                        if subF.__contains__("Peligro"):
                            lab = "Peligro"
                            numPeligro+=1
                        elif subF.__contains__("Prohibicion"):
                            lab = "Prohibicion"
                            numPro += 1
                        elif subF.__contains__("Stop"):
                            lab = "Stop"
                            numStop += 1
                        else:
                            lab = "Otros"
                            numOtros += 1
                        labels.append(diccionario.get(lab))
                        windowArray.append(window.ravel())
                        # cv2.imshow('img', window)
                        # cv2.imshow('img2', windowEXT)
                        # cv2.waitKey()
                        # cv2.destroyAllWindows()
                        if area2[0]<0 or area2[1]<0 or area2[2]>shape[0] or area2[3]>shape[1]:
                            windowArray10.append(window.ravel())
                        else:
                            windowEXT = Icopy[area2[1]:area2[1] + area2[3], area2[0]:area2[0] + area2[2]]
                            windowEXT = cv2.resize(windowEXT, (25, 25), None, 0, 0, cv2.INTER_NEAREST)
                            windowArray10.append(windowEXT.ravel())
                        count+=1
                    listaImagenes.append(subF+"\\"+filename)
                        # if count == 1:
                        #     return
                        # count+=1
                        # windowArray.append(windowEXT)
                        # cv2.rectangle(Icopy, (max[0], max[1]), (max[2], max[3]), (0, 255, 0), 2)
                        # cv2.imshow('img', Icopy)
                        # cv2.waitKey(0)
                        # cv2.destroyWindow("img")
    print(count)
    cantidadOtros, cantidadStop, cantidadPeligro, cantidadPro = 0, 0, 0, 0
    tamLista = listaImagenes.__len__()
    numImagenesPorCierto = int(np.round(tamLista*0.2))+1
    for i in range (0,tamLista):
        randInt = np.random.randint(0, tamLista)
        imagenRandom = listaImagenes[randInt]
        if(len(res)==numImagenesPorCierto):
            break
        if not res.__contains__(imagenRandom):
            res.append(imagenRandom)
        else:
            randInt = np.random.randint(0, numImagenesPorCierto)
            imagenRandom = listaImagenes[randInt]
            if not res.__contains__(imagenRandom):
                res.append(imagenRandom)
    print()
train()
print("----------------------")
LDA()