import os

import cv2
import numpy as np

from auxiliar_clases import graphic_features as graph
from auxiliar_clases import mathFunctions as Vect

mser = cv2.MSER_create()
mserTrain = cv2.MSER_create()
ImageDir = "Images_Sign_Detection_Benchmark"
# ImageDir = "Images_Sign_Recognition_Benchmark"
TRAIN_DIR = "./Training_Images/Training/" + ImageDir
#test_dir = './Final_Test/Images'
test_dir = './TEST_Calle'
test_ext = ('.jpg', '.ppm')
listDescriptor = []
listAreas=[]


def obtenerRegion(image_path):
    print("Test, processing ", image_path, "\n")
    I = cv2.imread(image_path)
    trainShape = I.shape
    print(trainShape)
    imageShape = [trainShape[1], trainShape[0]]
    Icopy = I.copy()
    Igray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

    areas = []

    minArea = int((imageShape[0] * imageShape[1]) / 2)
    # mser.setDelta(2)
    #mser.setMinArea(50)

    mid = None
    midT = False
    difShape = np.abs((trainShape[0] - trainShape[1]))
    if (difShape < 2):
        mid = (int(trainShape[0] / 2), int(trainShape[1] / 2))
        mser.setMinArea(minArea)
        midT = True
    regionsDetected = mser.detectRegions(Igray, None)
    # rects = [cv2.boundingRect(p.reshape(-1,1,2)) for p in regionsDetected]
    rects = []
    puntos = []
    aratio = 0
    areaSuficiente = False
    xS, yS, wS, hS = 0, 0, 0, 0
    Icopy2 = Icopy.copy()
    # print(len(regionsDetected))
    for p in regionsDetected:
        rect = cv2.boundingRect(p.reshape(-1, 1, 2))
        x, y, w, h = rect
        if (float(w) / float(h) > 1.2) or (float(w) / float(h) < 0.8):
            continue

        if (not Vect.contienePunto(rects, rect)):

            xS, yS, wS, hS = x, y, w, h
            if (midT):
                midX = int(np.abs(wS - xS))
                midY = int(np.abs(hS - yS))
                # rango = [mid[0]-10,mid[0]+10,mid[1]-10,mid[1]+10]
                # if(midX>rango[0] and midX<rango[1] and midY>rango[2] and midY<rango[3]):
                nuevaImagen = I[yS:yS + hS, xS:xS + wS]
                cv2.rectangle(Icopy, (x, y), (x + w, y + h), (0, 255, 0), thickness=1)
                Icopy = cv2.resize(Icopy, (imageShape[0] * 2, imageShape[1] * 2))
                cv2.imshow("area", Icopy)
                cv2.waitKey(1500)
                cv2.destroyWindow("area")
                Icopy = I.copy()
                graph.imitacionHOG(nuevaImagen)

            else:
                print("imagenes grandes")
                nuevaImagen = I[yS:yS + hS, xS:xS + wS]
                cv2.rectangle(Icopy, (x, y), (x + w, y + h), (0, 255, 0), thickness=1)
                Icopy = cv2.resize(Icopy, (imageShape[0], imageShape[1]))
                cv2.imshow("area", nuevaImagen)
                cv2.waitKey()
                cv2.destroyWindow("area")
                Icopy = I.copy()
            ## pasar HOG a la nuevaImagen y si pasa el filtro de los colores, buscar lineas/circulos
            # cv2.imshow("original", I)
            # cv2.imshow("nuevaImagen", nuevaImagen)
            # cv2.waitKey(200)
            # cv2.destroyAllWindows()
            # graph.lineas(nuevaImagen)
            rects.append(rect)

def train ():
    indice = 0
    regions = []
    for folder in os.listdir(TRAIN_DIR):
        parcial_path = os.path.join(TRAIN_DIR, folder)
        if (parcial_path.endswith(".ppm")):
            print("imagenes Sueltas -> ", parcial_path)

            region = obtenerRegion(parcial_path)
            regions.append(region)
            continue
        elif (parcial_path.endswith(".txt")):
            continue
        for filename in os.listdir(parcial_path):
            if os.path.splitext(filename)[1].lower() in test_ext:
                full_path = os.path.join(parcial_path, filename)

                region = obtenerRegion(full_path)
                regions.append(region)

# usoHOG()

train()
