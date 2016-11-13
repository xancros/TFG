import os

import OCR
import cv2
import numpy as np
from auxiliar_clases import graphic_features as graph
from auxiliar_clases import mathFunctions as Vect
from termcolor import colored

mser = cv2.MSER_create()
mserTrain = cv2.MSER_create()
ImageDir = "Images_Sign_Detection_Benchmark"
# ImageDir = "Images_Sign_Recognition_Benchmark"
TRAIN_DIR = "./Training_Images/Training/" + ImageDir
#test_dir = './Final_Test/Images'
test_ext = ('.jpg', '.ppm')


def checkAcumulator(index):
    if (len(index) > 1):
        if (index.__contains__(3)):
            index.remove(3)
            return checkAcumulator(index)
    return index[0]


def obtenerRegion(image_path):

    path = image_path.split("\\")
    I = cv2.imread(image_path)
    trainShape = I.shape
    Icopy = I.copy()
    Igray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    areas = np.zeros(shape=(3))
    regionsDetected = mserTrain.detectRegions(Igray, None)
    rects = []
    Icopy2 = Icopy.copy()
    listShapes = []
    listImages = []
    # if (image_path.__contains__("\\14\\00004.ppm")):
    #     cv2.imshow("original", I)
    #     cv2.waitKey(800)
    #     cv2.destroyAllWindows()
    # elif (image_path.__contains__("\\15\\")):
    #     raise Exception("FIn carpeta 14")
    # else:
    #     return "circle"

    for p in regionsDetected:

        rect = cv2.boundingRect(p.reshape(-1, 1, 2))

        x, y, w, h = rect
        if (float(w) / float(h) > 1.2) or (float(w) / float(h) < 0.8):
            continue

        if (not Vect.contienePunto(rects, rect)):

            xS, yS, wS, hS = x, y, w, h
            nuevaImagen = I[yS:yS + hS, xS:xS + wS]
            cv2.rectangle(Icopy, (x, y), (x + w, y + h), (0, 255, 0), thickness=1)
            Icopy = cv2.resize(Icopy, (200, 200), None, 0, 0, cv2.INTER_LANCZOS4)
            nuevaImagen = cv2.resize(nuevaImagen, (200, 200), None, 0, 0, cv2.INTER_LANCZOS4)
            # cv2.imshow("nueva imagen",nuevaImagen)
            # cv2.waitKey(1000)
            # cv2.destroyAllWindows()

            CircleShape, res = graph.shapeDetection(nuevaImagen, image_path)

            if (res == "circle"):
                # print("circle")
                listShapes.append(CircleShape)
                # res = trainAndTest(image2,overwrite=False)
                listImages.append(nuevaImagen)
                areas[0] += 1.05
                # break
            elif (res == "triangle"):
                # print("triangle")
                listShapes.append(CircleShape)
                listImages.append(nuevaImagen)
                areas[1] += 1.05
                # break
            else:
                # print("other")
                areas[2] += 0.1
            Icopy = I.copy()

            # else:
            #     a = 0
                # print("imagen grande")
            rects.append(rect)
    maxValue = np.amax(areas)

    index = np.where(areas == maxValue)[0]
    a = []
    if (maxValue == 0.0):
        return "background"
    index = checkAcumulator(index)
    if (index == 2):
        return "background"
    else:
        sortedList = sorted(listShapes, key=lambda elem: elem[1])
        bigger = sortedList[-1]
        targetShape = bigger[0]
        indexImage = listShapes.index(bigger)
        targetImage = listImages[indexImage]
        ocrResult = OCR.trainAndTest(targetImage, targetShape, overwrite=False)
        if (ocrResult == "stop"):
            return "background"
        if (index == 0):
            return "circle"
        else:
            return "triangle"
            # if (index == 0):
            #     # bigger = [[c1[0], c1[1], c4[0], c4[1]],r]
            #     # circ = sorted(circ, key=lambda c: c[2])
            #     # bigger = circ[-1]
            #     sortedList = sorted(listShapes,key=lambda elem:elem[1])
            #     bigger = sortedList[-1]
            #     targetShape=bigger[0]
            #     ocrResult = OCR.trainAndTest(nuevaImagen, targetShape, overwrite=False)
            #     if (ocrResult == "stop"):
            #         return "background"
            #     return "circle"
            # elif (index == 1):
            #     return "triangle"
            # else:
            #     return "background"


def train ():
    NImages = 0
    regions = []
    bgs = []
    for folder in os.listdir(TRAIN_DIR):
        parcial_path = os.path.join(TRAIN_DIR, folder)
        if (parcial_path.endswith(".ppm")):
            continue
        elif (parcial_path.endswith(".txt")):
            continue
        for filename in os.listdir(parcial_path):
            if os.path.splitext(filename)[1].lower() in test_ext:
                NImages += 1
                full_path = os.path.join(parcial_path, filename)

                print("Test, processing ", full_path, "\n")
                image = cv2.imread(full_path)
                image = cv2.resize(image, (200, 200), None, 0, 0, cv2.INTER_LANCZOS4)
                region = obtenerRegion(full_path)
                regions.append(region)
                path = full_path.split("\\")
                folderNumber = int(path[-2])
                if (region == "circle"):
                    s = ("THE IMAGE -> " + path[-2] + "//" + path[-1] + "// -> IS A CIRCLE SIGNAL")
                    print(colored(s, 'red'))
                elif (region == "triangle"):
                    s = ("THE IMAGE -> " + path[-2] + "//" + path[-1] + "// -> IS A DANGER (TRIANGLE) SIGNAL")
                    print(colored(s, 'green'))
                else:
                    s = ("THE IMAGE -> " + path[-2] + "//" + path[-1] + "// -> IS A BACKGROUND IMAGE")
                    print(colored(s, 'blue'))
                    bgs.append(image)
                    cv2.imshow("IMAGE", image)
                    cv2.waitKey(3000)
                    cv2.destroyAllWindows()
            if (folderNumber >= 14):
                cv2.imshow("IMAGE", image)
                cv2.waitKey(500)
                cv2.destroyAllWindows()
            if (full_path.__contains__("\\13\\")):
                print()
    print("END")
# usoHOG()

train()
