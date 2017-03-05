# Author: Jose Miguel Buenaposada 2015.
# Simple MSER application for traffic sign window proposal generation
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os,inspect
from shutil import copy as cp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
np.set_printoptions(suppress=True)
mser = cv2.MSER_create()
c0=0.99
PRIORS = np.array([((1-c0)/3), ((1-c0)/3), ((1-c0)/3), c0])
PRIORS /= PRIORS.sum()
scala_shape=(25,25)
#scala_shape=(50,50)
script_directory = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
test_dir = script_directory+"/Imagenes Deteccion/test/"
TRAIN_DIR = script_directory+"/Imagenes Deteccion/train/"
VALIDACION_DIR = script_directory+"/Validacion Cruzada/"
test_ext = ('.jpg', '.ppm')
diccionario = {'Peligro':0,'Prohibicion':1,'Stop':2,'Otros':3}
windowArray = []
windowArray10 = []
labels = []
labelsLow = []
labelsHigh = []
clasificadorLower = lda()
# clasificadorLower = lda()
clasificadorHigher = lda()
res = []
CLL = None
CLU = None

def printAreaTest():
    area2 = []
    for filename in os.listdir(test_dir):
        if os.path.splitext(filename)[1].lower() in test_ext:
            full_path = os.path.join(test_dir, filename)
            I = cv2.imread(full_path)
            Icopy = I.copy()
            Igray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
            shape = Icopy.shape
            # version opencv 3.1
            # regions = mser.detectRegions(Igray, None)
            regions = mser.detectRegions(Igray)
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
                    cv2.rectangle(Icopy, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.rectangle(Icopy,(pX1,pY1),(pX2,pY2),(255,0,0),2)
                    cv2.circle(Icopy,(pX1,pY1),4,(0,0,255),-1)
                    cv2.circle(Icopy, (pX2, pY1), 4, (0, 255, 0), -1)
                    cv2.circle(Icopy, (pX1, pY2), 4, (255, 0, 0), -1)
                    cv2.circle(Icopy, (pX2, pY2), 4, (0, 255, 255), -1)
                    cv2.imshow("area",area)
                    cv2.imshow("area2",area2)
                    cv2.imshow("imagen",Icopy)
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
                # version opencv 3.1
                # regions = mser.detectRegions(Igray, None)
                regions = mser.detectRegions(Igray)
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
                    senal = cv2.resize(areaTest, scala_shape, None, 0, 0, cv2.INTER_NEAREST)
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
                    labClase = "Peligro"
                elif clase == 1:
                    labClase = "Prohibicion"
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
    global labelsLow
    global labelsHigh
    global clasificadorLower
    global clasificadorHigher
    global CLL
    global CLU
    global PRIORS
    # clasificadorLower.set_params(priors=PRIORS)
    lowerWindow = np.vstack(windowArray)
    upperWindow = np.vstack(windowArray10)
    E = np.array(labels)
    ELow = np.array(labelsLow)
    EHigh = np.array(labelsHigh)
    # clasificadorLower.fit(lowerWindow,E)
    vectorCLower=clasificadorLower.fit_transform(lowerWindow,ELow)
    vectorCUpper=clasificadorHigher.fit_transform(upperWindow,EHigh)
    CRU = vectorCUpper.astype(np.float32,copy=True)
    CLU = cv2.ml.NormalBayesClassifier_create()
    CLU.train(CRU,cv2.ml.ROW_SAMPLE,EHigh)
    CRL = vectorCLower.astype(np.float32, copy=True)
    CLL = cv2.ml.NormalBayesClassifier_create()
    CLL.train(CRL, cv2.ml.ROW_SAMPLE, ELow)
    print()

    print()


def evaluarImagen(rutaImagen):
    global windowArray
    global windowArray10
    global labels
    global clasificadorLower
    global clasificadorHigher
    global CLL
    global CLU
    ###########################
    probabilidades = []
    path = (script_directory + "/salidaLower.txt")
    file = open(path, 'w')
    path2 = (script_directory + "/salidaHigher.txt")
    pathPrueba = (script_directory + "/misalida.txt")
    fileH = open(path2, 'w')
    fileP = open(pathPrueba, "w")
    contadorU, contadorL = 0, 0
    imagen = cv2.imread(rutaImagen)
    Icopy = imagen.copy()
    Icopy2 = Icopy.copy()
    Igray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    shape = Icopy.shape
    # version opencv 3.1
    # regions = mser.detectRegions(Igray, None)
    regions, _ = mser.detectRegions(Igray)
    rects = [cv2.boundingRect(p.reshape(-1, 1, 2)) for p in regions]
    for r in rects:
        x, y, w, h = r
        areaXYW = w * h
        # Simple aspect ratio filtering
        aratio = float(w) / float(h)
        if (aratio > 1.2) or (aratio < 0.8):
            continue
        areaTest = Icopy[y:y + h, x:x + w]
        senal = cv2.resize(areaTest, scala_shape, None, 0, 0, cv2.INTER_NEAREST)
        signal = senal.ravel()
        signalT = clasificadorLower.transform(signal)
        vectorSignalT = signalT.astype(np.float32)
        _, Yhat1, prob1 = CLU.predictProb(vectorSignalT)
        indice = Yhat1[0][0]
        if indice != 3:
            print(indice)
            cv2.rectangle(Icopy2, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("imagen", Icopy2)
    cv2.waitKey()
    cv2.destroyAllWindows()

def validacionCruzada():
    global windowArray
    global windowArray10
    global labels
    global clasificadorLower
    global clasificadorHigher
    global CLL
    global CLU
    ###########################
    probabilidades = []
    path = (script_directory + "/salidaLower.txt")
    file = open(path, 'w')
    path2 = (script_directory + "/salidaHigher.txt")
    pathPrueba = (script_directory+"/misalida.txt")
    fileH = open(path2, 'w')
    fileP=open(pathPrueba,"w")
    contadorU, contadorL = 0, 0
    for folder in os.listdir(VALIDACION_DIR):
        subF = os.path.join(VALIDACION_DIR, folder)
        for filename in os.listdir(subF):
            if os.path.splitext(filename)[1].lower() in test_ext:
                imagen = os.path.join(subF, filename)
                #imagen = "Z:\TFG\\URJC TFG/Validacion Cruzada/00000.ppm"
                test = cv2.imread(imagen)
                testRGB = cv2.cvtColor(test.copy(), cv2.COLOR_BGR2RGB)
                fileNameArr = imagen.split("/")
                fileName = fileNameArr[len(fileNameArr) - 1]
                fileName = fileName.replace("\\", "_")
                Icopy = test.copy()
                Icopy2 = test.copy()
                Igray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
                shape = Icopy.shape
                # version opencv 3.1
                # regions = mser.detectRegions(Igray, None)
                regions,_ = mser.detectRegions(Igray)
                rects = [cv2.boundingRect(p.reshape(-1, 1, 2)) for p in regions]
                print("Usando imagenes de train para test, processing " + imagen + "\n")
                fileP.write("Usando imagenes de train para test, processing " + imagen + "\n")
                file.write("Test, processing " + imagen + "\n")
                fileH.write("Test, processing " + imagen + "\n")
                areaFinal = [-1,-1,-1,-1]
                area = -1
                otros, peligro, stop, prohibicion = 0, 0, 0, 0
                for r in rects:
                    x, y, w, h = r
                    areaXYW=w*h
                    # Simple aspect ratio filtering
                    aratio = float(w) / float(h)
                    if (aratio > 1.2) or (aratio < 0.8):
                        continue
                    areaTest = Icopy[y:y + h, x:x + w]
                    # cv2.imshow("imagen", test)
                    # cv2.imshow("area",areaTest)
                    # cv2.waitKey(1200)
                    # cv2.destroyAllWindows()
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
                    senal = cv2.resize(areaTest, scala_shape, None, 0, 0, cv2.INTER_NEAREST)
                    signal = senal.ravel()
                    signalT = clasificadorLower.transform(signal)
                    vectorSignalT = signalT.astype(np.float32)
                    _, Yhat1, prob1 = CLL.predictProb(vectorSignalT)
                    senal2 = cv2.resize(areaTest,(80,80),None,0,0,cv2.INTER_NEAREST)
                    signal2 = senal2.ravel()
                    signalT2 = clasificadorHigher.transform(signal2)
                    vectorSignalT2 = signalT2.astype(np.float32)
                    _,Yhat2,prob2= CLU.predictProb(vectorSignalT2)
                    probabilidades.append(prob1)
                    indice = Yhat2[0][0]
                    # cv2.rectangle(Icopy2, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    ##tengo que cambiar esto para que no salgan tantas lineas, sino que cuente cuantas de clase 0,1,2,3 y 4 lineas nada mas
                    if indice == 3:
                        otros += 1
                    elif indice == 2:
                        stop += 1
                    elif indice == 0:
                        peligro += 1
                    elif indice == 1:
                        prohibicion += 1
                    if area < areaXYW:
                        area = areaXYW
                        areaFinal=[x,y,w,h]
                    if indice != 3 :
                        print(indice)
                        cv2.rectangle(Icopy2, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        # cv2.imshow("imagen", Icopy2)
                        # cv2.waitKey(800)
                        # cv2.destroyAllWindows()
                        # Icopy2=test.copy()
                supuestaClase = "La imagen <<" + fileName + ">>tendria que ser categorizada como "
                print("La imagen <<" + imagen + ">>tendría que ser categorizada como ")
                fileP.write("La imagen <<" + imagen + ">>tendría que ser categorizada como ")
                if imagen.__contains__("3-Otros"):
                    print("FONDO \n")
                    fileP.write("FONDO \n")
                    supuestaClase+="FONDO \n"
                elif imagen.__contains__("2-Stop"):
                    print("STOP \n")
                    fileP.write("STOP \n")
                    supuestaClase += "STOP \n"
                elif imagen.__contains__("1-Prohibicion"):
                    print("PROHIBICION \n")
                    fileP.write("PROHIBICION \n")
                    supuestaClase += "PROHIBICION \n"
                elif imagen.__contains__("0-Peligro"):
                    print("PELIGRO \n")
                    fileP.write("PELIGRO \n")
                    supuestaClase += "PELIGRO \n"
                print("Los resultados del clasificador son: \n")
                fileP.write("Los resultados del clasificador son: \n")
                print("FONDO -> " + str(otros) + "\n")
                fileP.write("FONDO -> " + str(otros) + "\n")
                print("STOP -> " + str(stop) + "\n")
                fileP.write("STOP -> " + str(stop) + "\n")
                print("PROHIBICION -> " + str(prohibicion) + "\n")
                fileP.write("PROHIBICION -> " + str(prohibicion) + "\n")
                print("PELIGRO -> " + str(peligro) + "\n")
                fileP.write("PELIGRO -> " + str(peligro) + "\n")
                ###########################
                clases = [peligro,prohibicion, stop, otros]
                maximo = np.amax(clases)
                clase = np.where(clases == maximo)[0][0]
                labClase = ""
                if clase == 0:
                    labClase = "Peligro"
                elif clase == 1:
                    labClase = "Prohibicion"
                elif clase == 2:
                    labClase = "Stop"
                elif clase == 3:
                    labClase = "Otros"

                # cv2.imshow("imagen", Icopy2)
                # cv2.waitKey()
                # cv2.destroyAllWindows()
                coord = [x, y, (x + w), (y + h)]
                coordenadas = (
                "X:" + str(areaFinal[0]) + " Y:" + str(areaFinal[1]) + " X2:" + str(areaFinal[2]) + " Y2:" + str(areaFinal[3]))
                file.write(supuestaClase)
                file.write("Clase supuesta -> " + str(clase) + "(" + labClase + ")" + "\n")
                if not labClase.__eq__("Otros"):
                    file.write("Coordenadas = " + coordenadas + "\n")
                file.write("--------------------------------------" + "\n")
                file.write("Cuanta cantidad de Fondo se reconoce en la imagen ->" + str(otros) + "\n")
                file.write("Cuanta cantidad de Peligro se reconoce en la imagen ->" + str(peligro) + "\n")
                file.write("Cuanta cantidad de Prohibicion se reconoce en la imagen ->" + str(prohibicion) + "\n")
                file.write("Cuanta cantidad de Stop se reconoce en la imagen ->" + str(stop) + "\n")
                file.write("--------------------------------------" + "\n")
                otros, peligro, stop, prohibicion = 0, 0, 0, 0
                senal = cv2.resize(area2, scala_shape, None, 0, 0, cv2.INTER_NEAREST)
                # signal = (np.asarray(senal)).reshape(1, -1)
                signal = senal.ravel()
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
    global labelsLow
    global labelsHigh
    count = 0
    count2 = 0
    listaImagenes = []
    mas50 = 0
    menos50 = 0
    # path = (script_directory + "/shapes.txt")
    # file = open(path, 'w')
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
                    # full_path="Z:\TFG\\URJC TFG/Validacion Cruzada/00000.ppm"
                    I = cv2.imread(full_path)
                    Icopy = I.copy()
                    Igray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
                    shape = Icopy.shape
                    imagen = cv2.imread(full_path,0)
                    # version opencv 3.1
                    # regions = mser.detectRegions(Igray, None)
                    regions,_ = mser.detectRegions(Igray)
                    rects = [cv2.boundingRect(p.reshape(-1, 1, 2)) for p in regions]
                    for r in rects:
                        x, y, w, h = r
                        # Simple aspect ratio filtering
                        aratio = float(w) / float(h)
                        if (aratio > 1.2) or (aratio < 0.8):
                            continue

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
                        if I.shape[0] > 50 and I.shape[1] > 50:
                            window = cv2.resize(window, scala_shape, None, 0, 0, cv2.INTER_NEAREST)
                            windowArray.append(window.ravel())
                            labelsLow.append(diccionario.get(lab))
                        else:
                            window = cv2.resize(window, (80,80), None, 0, 0, cv2.INTER_NEAREST)
                            if area2[0]<0 or area2[1]<0 or area2[2]>shape[0] or area2[3]>shape[1]:
                                windowArray10.append(window.ravel())
                            else:
                                windowEXT = Icopy[area2[1]:area2[1] + area2[3], area2[0]:area2[0] + area2[2]]
                                windowEXT = cv2.resize(windowEXT, (80,80), None, 0, 0, cv2.INTER_NEAREST)
                                windowArray10.append(windowEXT.ravel())
                            labelsHigh.append(diccionario.get(lab))
                        # cv2.imshow('img', window)
                        #
                        # cv2.waitKey(800)
                        # cv2.destroyAllWindows()
                        # if area2[0]<0 or area2[1]<0 or area2[2]>shape[0] or area2[3]>shape[1]:
                        #     windowArray10.append(window.ravel())
                        # else:
                        #     windowEXT = Icopy[area2[1]:area2[1] + area2[3], area2[0]:area2[0] + area2[2]]
                        #     windowEXT = cv2.resize(windowEXT, scala_shape, None, 0, 0, cv2.INTER_NEAREST)
                        #     windowArray10.append(windowEXT.ravel())
                        if(I.shape[0]>50 and I.shape[1]>50):
                            mas50+=1
                        elif (I.shape[0]<=50 and I.shape[1]<=50):
                            menos50+=1
                        # file.write(full_path+"\n"+str(I.shape)+"\n")

                        count+=1
                    listaImagenes.append(subF+"\\"+filename)
    # file.write("imagenes con shape > 50 = "+str(mas50))
    # file.write(" imagenes con shape <= 50 = " + str(menos50))
    # file.close()
                        # if count == 1:
                        #     return
                        # count+=1
                        # windowArray.append(windowEXT)
                        # cv2.rectangle(Icopy, (max[0], max[1]), (max[2], max[3]), (0, 255, 0), 2)
                        # cv2.imshow('img', Icopy)
                        # cv2.waitKey(0)
                        # cv2.destroyWindow("img")
    # print(count)
    # cantidadOtros, cantidadStop, cantidadPeligro, cantidadPro = 0, 0, 0, 0
    # tamLista = listaImagenes.__len__()
    # numImagenesPorCierto = int(np.round(tamLista*0.2))+1
    # for i in range (0,tamLista):
    #     randInt = np.random.randint(0, tamLista)
    #     imagenRandom = listaImagenes[randInt]
    #     if(len(res)==numImagenesPorCierto):
    #         break
    #     if not res.__contains__(imagenRandom):
    #         res.append(imagenRandom)
    #         copiar(imagenRandom)
    #     else:
    #         randInt = np.random.randint(0, numImagenesPorCierto)
    #         imagenRandom = listaImagenes[randInt]
    #         if not res.__contains__(imagenRandom):
    #             res.append(imagenRandom)
    #             copiar(imagenRandom)
    # print()

def copiar(pathFileSource):
    dst=""
    if pathFileSource.__contains__("Peligro"):
        dst="0-Peligro"
    elif pathFileSource.__contains__("Prohibicion"):
        dst="1-Prohibicion"
    elif pathFileSource.__contains__("Stop"):
        dst="2-Stop"
    else:
        dst="3-Otros"
    cp(pathFileSource,"./TODAS/"+dst)



# printAreaTest()
# print()
train()

# print("----------------------")
LDA()
# print("----------------------")
# testTrain()
validacionCruzada()