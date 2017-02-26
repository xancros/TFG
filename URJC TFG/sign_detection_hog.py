import cv2
import numpy as np
import matplotlib.pyplot as plt
import os,inspect
from skimage.feature import hog
from skimage import data, color, exposure
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
np.set_printoptions(suppress=True)
mser = cv2.MSER_create()
c0=0.7
PRIORS = np.array([c0, ((1-c0)/3), ((1-c0)/3), ((1-c0)/3)])
PRIORS /= PRIORS.sum()
scala_shape=(25,25)
scala_shape=(50,50)
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

def LDA():
    global windowArray
    global windowArray10
    global labels
    global clasificadorLower
    global clasificadorHigher
    global CLL
    global CLU
    global PRIORS
    clasificadorLower.set_params(priors=PRIORS)
    lowerWindow = np.vstack(windowArray)
    upperWindow = np.vstack(windowArray10)
    E = np.array(labels)
    # clasificadorLower.fit(lowerWindow,E)
    vectorCLower=clasificadorLower.fit_transform(lowerWindow,E)
    vectorCUpper=clasificadorHigher.fit_transform(upperWindow,E)
    CRU = vectorCUpper.astype(np.float32,copy=True)
    CLU = cv2.ml.NormalBayesClassifier_create()
    CLU.train(CRU,cv2.ml.ROW_SAMPLE,E)
    CRL = vectorCLower.astype(np.float32, copy=True)
    CLL = cv2.ml.NormalBayesClassifier_create()
    CLL.train(CRL, cv2.ml.ROW_SAMPLE, E)


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
        imagen = "Z:\TFG\\URJC TFG/Imagenes Deteccion/train/3-Otros\\38\\00047.ppm"
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
        senal = cv2.resize(area2, scala_shape, None, 0, 0, cv2.INTER_NEAREST)
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
                    regions,_ = mser.detectRegions(Igray)
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
                        window = cv2.resize(window, (200,200), None, 0, 0, cv2.INTER_LANCZOS4)
                        window = HOG(window)
                        window = cv2.resize(window, scala_shape, None, 0, 0, cv2.INTER_NEAREST)
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
                        ##LLAMADA HOG

                        ##
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
                            windowEXT = cv2.resize(windowEXT, scala_shape, None, 0, 0, cv2.INTER_NEAREST)
                            ##LLAMADA HOG
                            windowEXT = HOG(windowEXT)
                            ##
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


def HOG (window):
    image = cv2.cvtColor(window,cv2.COLOR_BGR2GRAY)
    fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualise=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Input image')
    ax1.set_adjustable('box-forced')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    ax1.set_adjustable('box-forced')
    plt.show()

    ######
    return hog_image

train()