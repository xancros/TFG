import cv2
import numpy as np
import matplotlib as plt
import os
from termcolor import colored
class sign_detection:
    TRAIN_DIR=""
    TEST_DIR=""
    FLANN_INDEX_KDTREE=0
    index_params=dict()
    search_params=dict()
    IMG_FORMAT=('.jpg','.ppm')
    listDescriptor=[]

    def __init__(self):
        listDescriptorTrain = []
        TRAIN_DIR = ""
        TEST_DIR = ""
        index_params = dict()
        search_params = dict()
        IMG_FORMAT = ('.jpg', '.ppm')
        FLANN_INDEX_KDTREE = 3
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary
        TRAIN_DIR="./Final_Training/Images"
        TEST_DIR="C:/Users/xancr/Documents/Final_Test/Images"

    #Detector MSER
    mser = cv2.MSER_create()
    mserTrain = cv2.MSER_create()
    #Descriptor FLANN

    flann = cv2.FlannBasedMatcher(indexParams=index_params,searchParams=search_params)
    #####################

    #Descriptor BF

    bf = cv2.BFMatcher(normType=cv2.NORM_HAMMING)
    #####################

    def containsPoint(points,point):
        if(len(points)>0):
            r=points[-1]
            x,y,w,h=points
            xR,yR,wR,hR=r
            if(x==xR and y==yR and w==wR and h==hR):
                return True
        return False
    def setMser(mser,delta=2, minArea=200):
        mser.setDelta(delta)
        mser.setMinArea(minArea)
    def getRegions(self):

        indice = 0
        for folder in os.listdir(sign_detection.TRAIN_DIR):
            parcial_path = os.path.join(sign_detection.TRAIN_DIR, folder)
            for filename in os.listdir(parcial_path):
                if os.path.splitext(filename)[1].lower() in sign_detection.IMG_FORMAT:
                    print("Test, processing ", filename, "\n")
                    full_path = os.path.join(parcial_path, filename)
                    regions = []
                    I = cv2.imread(full_path)
                    trainShape = I.shape
                    print(trainShape)
                    imageShape = (trainShape[1], trainShape[0])
                    Icopy = I.copy()
                    Igray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
                    sign_detection.mserTrain.setDelta(2)
                    sign_detection.mserTrain.setMinArea(200)
                    regionsDetected = sign_detection.mserTrain.detectRegions(Igray, None)
                    # rects = [cv2.boundingRect(p.reshape(-1,1,2)) for p in regionsDetected]
                    rects = []
                    areas = []
                    puntos = []
                    aratio = 0
                    Icopy2 = Icopy.copy()
                    for p in regionsDetected:
                        rect = cv2.boundingRect(p.reshape(-1, 1, 2))
                        x, y, w, h = rect
                        # hay que hacer un filtro como el de aspect ratio pero con la aparencia de los puntos(x,y,w,h)

                        # Simple aspect ratio filtering
                        if (float(w) / float(h) > 1.2) or (float(w) / float(h) < 0.8):
                            continue
                        if (not sign_detection.containsPoint(puntos, rect)):
                            if (w == h):
                                if (w - x >= 30 and h - y >= 30):
                                    print(colored("El area es suficientemente grande", 'green'))
                                    cv2.rectangle(Icopy, (x, y), (x + w, y + h), (0, 255, 0), thickness=1)
                                else:
                                    print(colored("El area seleccionada es la imagen", 'red'))
                                    cv2.rectangle(Icopy, (0, 0), (trainShape[0], trainShape[1]), (0, 255, 0),
                                                  thickness=1)

                                print(rect)

                                cv2.imshow("area", Icopy)
                                cv2.waitKey(1000)
                                cv2.destroyWindow("area")
                                Icopy = Icopy2.copy()
                                rects.append(rect)
