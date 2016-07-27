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
    path = image_path.split("\\")
    print("//" + path[-2] + "//" + path[-1])
    I = cv2.imread(image_path)
    trainShape = I.shape

    imageShape = [trainShape[1], trainShape[0]]
    Icopy = I.copy()
    Igray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

    areas = np.zeros(shape=(3))

    minArea = int((imageShape[0] * imageShape[1]) / 2)

    regionsDetected = mser.detectRegions(Igray, None)
    # rects = [cv2.boundingRect(p.reshape(-1,1,2)) for p in regionsDetected]
    rects = []
    puntos = []
    aratio = 0
    areaSuficiente = False
    xS, yS, wS, hS = 0, 0, 0, 0
    Icopy2 = Icopy.copy()
    print(len(regionsDetected))
    # if (image_path.__contains__("\\11\\")):
    #     print()
    #     cv2.imshow("original", I)
    #     cv2.waitKey(800)
    #     cv2.destroyAllWindows()
    # else:
    #     return
    for p in regionsDetected:

        rect = cv2.boundingRect(p.reshape(-1, 1, 2))

        x, y, w, h = rect
        if (float(w) / float(h) > 1.2) or (float(w) / float(h) < 0.8):
            continue

        if (not Vect.contienePunto(rects, rect)):

            xS, yS, wS, hS = x, y, w, h
            # if (midT):
            nuevaImagen = I[yS:yS + hS, xS:xS + wS]
            cv2.rectangle(Icopy, (x, y), (x + w, y + h), (0, 255, 0), thickness=1)
            Icopy = cv2.resize(Icopy, (200, 200), None, 0, 0, cv2.INTER_LANCZOS4)
            nuevaImagen = cv2.resize(nuevaImagen, (200, 200), None, 0, 0, cv2.INTER_LANCZOS4)
            # cv2.imshow("nueva imagen",nuevaImagen)
            # cv2.waitKey(1000)
            # cv2.destroyAllWindows()

            res = graph.shapeDetection(nuevaImagen, image_path)

            if (res == "circle"):
                print("circle")
                areas[0] += 1
                # break
            elif (res == "triangle"):
                print("triangle")
                areas[1] += 1
                # break
            else:
                print("other")
                areas[2] += 0.5
            Icopy = I.copy()

            # else:
            #     a = 0
                # print("imagen grande")
            rects.append(rect)
    maxValue = np.amax(areas)
    index = np.where(areas == maxValue)[0]
    if (index == 0):
        return "circle"
    elif (index == 1):
        return "triangle"
    else:
        return "background"
def train ():
    indice = 0
    regions = []
    for folder in os.listdir(TRAIN_DIR):
        parcial_path = os.path.join(TRAIN_DIR, folder)
        if (parcial_path.endswith(".ppm")):
            print("imagenes Sueltas -> ", parcial_path)
            #
            # region = obtenerRegion(parcial_path)
            # regions.append(region)
            continue
        elif (parcial_path.endswith(".txt")):
            continue
        for filename in os.listdir(parcial_path):
            if os.path.splitext(filename)[1].lower() in test_ext:
                full_path = os.path.join(parcial_path, filename)
                region = obtenerRegion(full_path)
                regions.append(region)
                path = full_path.split("\\")
                if (region == "circle"):
                    print("THE IMAGE -> " + path[-2] + "//" + path[-1] + "// -> IS A CIRCLE SIGNAL")
                elif (region == "triangle"):
                    print("THE IMAGE -> " + path[-2] + "//" + path[-1] + "// -> IS A DANGER (TRIANGLE) SIGNAL")
                else:
                    print("THE IMAGE -> " + path[-2] + "//" + path[-1] + "// -> IS A BACKGROUND IMAGE")


# usoHOG()

train()
