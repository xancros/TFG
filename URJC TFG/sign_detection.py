# Author: Jose Miguel Buenaposada 2015.
# Simple MSER application for traffic sign window proposal generation
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os,inspect
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
np.set_printoptions(suppress=True)
mser = cv2.MSER_create()

script_directory = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
test_dir = script_directory+"/Imagenes Deteccion/test/"
TRAIN_DIR = script_directory+"/Imagenes Deteccion/train/"
test_ext = ('.jpg', '.ppm')
diccionario = {'Prohibicion':0,'Peligro':1,'Stop':2,'Otros':3}
windowArray = []
windowArray10 = []
labels = []

clasificadorLower = lda()
clasificadorHigher = lda()
res = []
CLL = None
CLU = None

def printAreaTest():
    global windowArray
    global windowArray10
    global labels
    global clasificadorLower
    global clasificadorHigher
    global CLL
    global CLU
    area2 = []
    for filename in os.listdir(test_dir):
        if os.path.splitext(filename)[1].lower() in test_ext:
            full_path = os.path.join(test_dir, filename)
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
                area = I[y:y+h,x:x+w]
                x10,y10,w10,h10 = int(x*0.1),int(y*0.1),int(w*0.1),int(h*0.1)
                x2,y2,w2,h2 = x-x10,y-y10,w+w10,h+h10
                w2w=int(w2-w)

                pX1,pY1,pX2,pY2 = x2,y2,x+x10+w2,y+y10+h2
                # if(pX1<0):pX1=0
                # elif(pX1>shape[0]):pX1=shape[0]
                # if(pX2>shape[0]):pX2=shape[0]
                # elif(pX2<0):pX2=0
                # if(pY1<0):pY1=0
                # elif(pY1>shape[1]):pY1=shape[1]
                # if(pY2>shape[1]):pY2=shape[1]
                # elif(pY2<0):pY2=0
                area2 = I[pY1:pY2,pX1:pX2]
                lenArea = np.array(area2).size
                if(lenArea>0):
                    clase=""
                    senal = cv2.resize(area, (25, 25), None, 0, 0, cv2.INTER_NEAREST)
                    signal = (np.asarray(senal)).reshape(1, -1)
                    signalT = clasificadorLower.transform(signal)
                    vectorSignalT = signalT.astype(np.float32)
                    _, Yhat1, prob1 = CLL.predictProb(vectorSignalT)
                    labClase = ""
                    if clase == 0:
                        labClase = "Prohibicion"

                    elif clase == 1:
                        labClase = "Peligro"
                    elif clase == 2:
                        labClase = "Stop"
                    elif clase == 3:
                        labClase = "Otros"
                    cv2.rectangle(Icopy, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.rectangle(Icopy,(pX1,pY1),(pX2,pY2),(255,0,0),2)
                    cv2.circle(Icopy,(pX1,pY1),4,(0,0,255),-1)
                    cv2.circle(Icopy, (pX2, pY1), 4, (0, 255, 0), -1)
                    cv2.circle(Icopy, (pX1, pY2), 4, (255, 0, 0), -1)
                    cv2.circle(Icopy, (pX2, pY2), 4, (0, 255, 255), -1)
                    cv2.imshow("area",area)
                    cv2.imshow("area2",area2)
                    # cv2.imshow("imagen",Icopy)
                    cv2.waitKey(500)
                    cv2.destroyAllWindows()

                    Icopy=I.copy()
            cv2.imshow('img', Icopy)
            cv2.waitKey(0)

def testTrain():
        global windowArray
        global windowArray10
        global labels
        global clasificadorLower
        global clasificadorHigher
        global CLL
        global CLU
        path = (script_directory + "/pruebaTestLower.txt")
        file = open(path, 'w')
        for filename in os.listdir(test_dir):
            if os.path.splitext(filename)[1].lower() in test_ext:
                print ("Test, processing ", filename, "\n")
                full_path = os.path.join(test_dir, filename)
                imagen = full_path
                test = cv2.imread(imagen)
                testRGB = cv2.cvtColor(test.copy(), cv2.COLOR_BGR2RGB)
                fileNameArr = imagen.split("/")
                fileName = fileNameArr[len(fileNameArr) - 1]
                fileName = fileName.replace("\\", "_")
                Icopy = test.copy()
                Igray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
                shape = Icopy.shape
                regions = mser.detectRegions(Igray, None)
                rects = [cv2.boundingRect(p.reshape(-1, 1, 2)) for p in regions]
                print("Usando imagenes de train para test, processing " + imagen + "\n")
                file.write("Test, processing " + imagen + "\n")
                # fileH.write("Test, processing " + imagen + "\n")
                otros, peligro, stop, prohibicion = 0, 0, 0, 0
                for r in rects:
                    x, y, w, h = r
                    # Simple aspect ratio filtering
                    aratio = float(w) / float(h)
                    if (aratio > 1.2) or (aratio < 0.8):
                        continue
                    areaTest = Icopy[y:y + h, x:x + w]
                    # cv2.imshow("imagen", test)
                    # cv2.imshow("area",areaTest)
                    # cv2.waitKey(800)
                    # cv2.destroyWindow("area")
                    # # cv2.destroyAllWindows()
                    x1 = int(x * 0.1) + x
                    xw = x + w
                    x2 = int(xw * 0.1) + xw
                    y1 = int(y * 0.1) + y
                    yh = y + h
                    y2 = int(yh * 0.1) + yh
                    if x1 < 0 or y1 < 0 or x2 > shape[0] or y2 > shape[1]:
                        area2 = areaTest
                    else:
                        area2 = Icopy[y1:y1 + y2, x1:x1 + x2]
                    senal = cv2.resize(areaTest, (25, 25), None, 0, 0, cv2.INTER_NEAREST)
                    signal = (np.asarray(senal)).reshape(1, -1)
                    signalT = clasificadorLower.transform(signal)
                    vectorSignalT = signalT.astype(np.float32)
                    _, Yhat1, prob1 = CLL.predictProb(vectorSignalT)
                    ##tengo que cambiar esto para que no salgan tantas lineas, sino que cuente cuantas de clase 0,1,2,3 y 4 lineas nada mas
                    if imagen.__contains__("Otros"):
                        otros += 1
                    elif imagen.__contains__("Stop"):
                        stop += 1
                    elif imagen.__contains__("Prohibicion"):
                        prohibicion += 1
                    elif imagen.__contains__("Peligro"):
                        peligro += 1
                ###########################
                clases = [prohibicion, peligro, stop, otros]
                maximo = np.amax(clases)
                clase = np.where(clases == maximo)[0][0]
                labClase = ""
                if clase == 0:
                    labClase = "Prohibicion"

                elif clase == 1:
                    labClase = "Peligro"
                elif clase == 2:
                    labClase = "Stop"
                elif clase == 3:
                    labClase = "Otros"
                coord = [x, y, (x + w), (y + h)]
                coordenadas = (
                "X:" + str(coord[0]) + " Y:" + str(coord[1]) + " X2:" + str(coord[2]) + " Y2:" + str(coord[3]))
                file.write("Clase supuesta -> " + str(clase) + "(" + labClase + ")" + "\n")
                if not labClase.__eq__("Otros"):
                    file.write("Coordenadas = " + coordenadas + "\n")
                    cv2.imshow("imagen", test)
                    plt.imshow(testRGB, interpolation="bicubic")
                    plt.show()
                    cv2.waitKey()
                    cv2.destroyAllWindows()
                file.write("--------------------------------------" + "\n")
                file.write("Cuanta cantidad de Fondo se reconoce en la imagen ->" + str(otros) + "\n")
                file.write("Cuanta cantidad de Peligro se reconoce en la imagen ->" + str(peligro) + "\n")
                file.write("Cuanta cantidad de Prohibicion se reconoce en la imagen ->" + str(prohibicion) + "\n")
                file.write("Cuanta cantidad de Stop se reconoce en la imagen ->" + str(stop) + "\n")
                file.write("--------------------------------------" + "\n")
def LDA():
    global windowArray
    global windowArray10
    global labels
    global clasificadorLower
    global clasificadorHigher
    global CLL
    global CLU
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

    print()


def validacionCruzada():
    global windowArray
    global windowArray10
    global labels
    global clasificadorLower
    global clasificadorHigher
    global CLL
    global CLU
    ###########################
    path = (script_directory + "/salidaLower.txt")
    file = open(path, 'w')
    path2 = (script_directory + "/salidaHigher.txt")
    fileH = open(path2, 'w')
    contadorU,contadorL=0,0
    for imagen in res:

        test = cv2.imread(imagen)
        testRGB = cv2.cvtColor(test.copy(), cv2.COLOR_BGR2RGB)
        fileNameArr = imagen.split("/")
        fileName = fileNameArr[len(fileNameArr) - 1]
        fileName = fileName.replace("\\", "_")
        Icopy = test.copy()
        Igray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
        shape = Icopy.shape
        regions = mser.detectRegions(Igray, None)
        rects = [cv2.boundingRect(p.reshape(-1, 1, 2)) for p in regions]
        print("Usando imagenes de train para test, processing " + imagen + "\n")
        file.write("Test, processing " + imagen + "\n")
        fileH.write("Test, processing " + imagen + "\n")
        otros, peligro, stop, prohibicion = 0, 0, 0, 0
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
            # cv2.destroyAllWindows()
            x1 = int(x*0.1)+x
            xw = x+w
            x2 = int(xw*0.1)+xw
            y1 = int(y*0.1)+y
            yh = y+h
            y2 = int(yh*0.1)+yh
            if x1 < 0 or y1 < 0 or x2 > shape[0] or y2 > shape[1]:
                area2=areaTest
            else:
                area2 = Icopy[y1:y1+y2,x1:x1+x2]
            senal = cv2.resize(areaTest, (25, 25), None, 0, 0, cv2.INTER_NEAREST)
            signal = (np.asarray(senal)).reshape(1, -1)
            signalT = clasificadorLower.transform(signal)
            vectorSignalT = signalT.astype(np.float32)
            _, Yhat1, prob1 = CLL.predictProb(vectorSignalT)
            ##tengo que cambiar esto para que no salgan tantas lineas, sino que cuente cuantas de clase 0,1,2,3 y 4 lineas nada mas
            if imagen.__contains__("Otros"):
                otros += 1
            elif imagen.__contains__("Stop"):
                stop += 1
            elif imagen.__contains__("Prohibicion"):
                prohibicion += 1
            elif imagen.__contains__("Peligro"):
                peligro += 1
        ###########################
        clases = [prohibicion, peligro, stop, otros]
        maximo = np.amax(clases)
        clase = np.where(clases == maximo)[0][0]
        labClase = ""
        if clase == 0:
            labClase = "Prohibicion"

        elif clase == 1:
            labClase = "Peligro"
        elif clase == 2:
            labClase = "Stop"
        elif clase == 3:
            labClase = "Otros"
        coord = [x,y,(x+w),(y+h)]
        coordenadas = ("X:"+str(coord[0])+" Y:"+str(coord[1])+" X2:"+str(coord[2])+" Y2:"+str(coord[3]))
        file.write("Clase supuesta -> " + str(clase) + "(" + labClase + ")" + "\n")
        if not labClase.__eq__("Otros"):
            file.write ("Coordenadas = "+coordenadas+"\n")
        file.write("--------------------------------------" + "\n")
        file.write("Cuanta cantidad de Fondo se reconoce en la imagen ->" + str(otros) + "\n")
        file.write("Cuanta cantidad de Peligro se reconoce en la imagen ->" + str(peligro) + "\n")
        file.write("Cuanta cantidad de Prohibicion se reconoce en la imagen ->" + str(prohibicion) + "\n")
        file.write("Cuanta cantidad de Stop se reconoce en la imagen ->" + str(stop) + "\n")
        file.write("--------------------------------------" + "\n")
        otros, peligro, stop, prohibicion = 0, 0, 0, 0
        senal = cv2.resize(area2, (25, 25), None, 0, 0, cv2.INTER_NEAREST)
        signal = (np.asarray(senal)).reshape(1, -1)
        signalT = clasificadorLower.transform(signal)
        vectorSignalT = signalT.astype(np.float32)
        _, Yhat1, prob1 = CLL.predictProb(vectorSignalT)
        ##tengo que cambiar esto para que no salgan tantas lineas, sino que cuente cuantas de clase 0,1,2,3 y 4 lineas nada mas
        if imagen.__contains__("Otros"):
            otros += 1
        elif imagen.__contains__("Stop"):
            stop += 1
        elif imagen.__contains__("Prohibicion"):
            prohibicion += 1
        elif imagen.__contains__("Peligro"):
            peligro += 1
            ###########################
        clases = [prohibicion, peligro, stop, otros]
        maximo = np.amax(clases)
        clase = np.where(clases == maximo)[0][0]
        labClase = ""
        if clase == 0:
            labClase = "Prohibicion"
        elif clase == 1:
            labClase = "Peligro"
        elif clase == 2:
            labClase = "Stop"
        elif clase == 3:
            labClase = "Otros"

        fileH.write("Clase supuesta -> " + str(clase) + "(" + labClase + ")" + "\n")
        fileH.write("--------------------------------------" + "\n")
        fileH.write("Cuanta cantidad de Fondo se reconoce en la imagen ->" + str(otros) + "\n")
        fileH.write("Cuanta cantidad de Peligro se reconoce en la imagen ->" + str(peligro) + "\n")
        fileH.write("Cuanta cantidad de Prohibicion se reconoce en la imagen ->" + str(prohibicion) + "\n")
        fileH.write("Cuanta cantidad de Stop se reconoce en la imagen ->" + str(stop) + "\n")
        fileH.write("--------------------------------------" + "\n")
        # plt.imshow(testRGB, interpolation="bicubic")
        # plt.show()
        # plt.close()
        # cv2.imshow("TEST", test)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
    file.close()
    fileH.close()

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
printAreaTest()
# print()
# train()
# print("----------------------")
# LDA()
# print("----------------------")
# testTrain()
# validacionCruzada()