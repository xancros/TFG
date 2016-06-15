import math
import os

import cv2
import numpy as np

from auxiliar_clases import graphic_features as graph
from auxiliar_clases import mathFunctions as Vect

mser = cv2.MSER_create()
mserTrain = cv2.MSER_create()
TRAIN_DIR = "./Training_Images/Training/Images"
#test_dir = './Final_Test/Images'
test_dir = './TEST_Calle'
test_ext = ('.jpg', '.ppm')
listDescriptor = []
listAreas=[]
#Descriptor FLANN
FLANN_INDEX_KDTREE = 3
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(indexParams=index_params,searchParams=search_params)
#####################

#Descriptor BF

bf = cv2.BFMatcher(normType=cv2.NORM_HAMMING)


##############################################


def imitacionHOG(img):
    b, g, r = cv2.split(img)
    cv2.imshow("blue", b)
    cv2.imshow("green", g)
    cv2.imshow("red", r)
    maxB, maxG, maxR = np.amax(b), np.amax(g), np.amax(r)
    print(maxB, maxG, maxR)
    cv2.destroyAllWindows()
    list = np.asanyarray([maxB, maxG, maxR])
    index = np.where(list >= 200)[0]
    if (len(index) > 1):
        print("varios canales juegan")
        max = np.amax(list)
        indexMax = np.where(list == max)[0]
        if (len(indexMax) > 1):
            return -1
        return indexMax[0]
    elif (index[0] == 0):
        print("la imagen es muy azul")
        return 0
    elif (index[0] == 1):
        print("la imagen es muy verde")
        return -1
    else:
        print("la imagen es roja")
        return 2
    return img

def train ():
    indice =0
    for folder in os.listdir(TRAIN_DIR):
        parcial_path = os.path.join(TRAIN_DIR, folder)
        for filename in os.listdir(parcial_path):
            if os.path.splitext(filename)[1].lower() in test_ext:
                full_path = os.path.join(parcial_path, filename)
                print("Test, processing ", full_path, "\n")
                regions = []
                I = cv2.imread(full_path)
                I = cv2.imread("./auxiliar_images/velocidad.jpg")
                # lineas(I)
                # color = imitacionHOG(I)
                # colour = ""
                # if (color == 0):  # azul
                #     colour = "blue"
                # elif (color == 2):  # rojo
                #     colour = "red"
                # lineas(I)
                trainShape = I.shape
                # pruebaLineas(None,"otro")
                imageShape = (trainShape[1],trainShape[0])
                Icopy = I.copy()
                Igray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
                mser.setDelta(2)
                areas = []

                mser.setMinArea(50)
                regionsDetected = mser.detectRegions(Igray, None)
                # rects = [cv2.boundingRect(p.reshape(-1,1,2)) for p in regionsDetected]
                rects = []
                puntos = []
                aratio = 0
                areaSuficiente=False
                xS,yS,wS,hS=0,0,0,0
                Icopy2=Icopy.copy()
                print(len(regionsDetected))
                for p in regionsDetected:
                    rect = cv2.boundingRect(p.reshape(-1, 1, 2))
                    x, y, w, h = rect
                    if (float(w) / float(h) > 1.2) or (float(w) / float(h) < 0.8):
                        continue

                    if (not Vect.contienePunto(rects, rect)):
                        cv2.rectangle(Icopy, (x, y), (x + w, y + h), (0, 255, 0), thickness=1)
                        cv2.circle(Icopy, (int(x), int(y)), 3, (255, 0, 0), -1)
                        cv2.circle(Icopy, (int(x + w), int(h + y)), 3, (0, 255, 0), -1)
                        cv2.imshow("area", Icopy)
                        cv2.waitKey()
                        cv2.destroyWindow("area")


                        xS, yS, wS, hS = x, y, w, h
                        nuevaImagen = I[yS:yS + hS, xS:xS + wS]
                        ## pasar HOG a la nuevaImagen y si pasa el filtro de los colores, buscar lineas/circulos
                        # cv2.imshow("original", I)
                        # cv2.imshow("nuevaImagen", nuevaImagen)
                        # cv2.destroyAllWindows()
                        lineas(nuevaImagen)
                        rects.append(rect)


def lineas(imageName=None):
    if (isinstance(imageName, str)):
        if (imageName is None):
            fn = cv2.imread("./ceda.jpg")
        else:
            fn = cv2.imread(imageName)
    else:
        fn = imageName
    # src = cv2.imread(fn)
    src = fn.copy()
    imShape = fn.shape
    bn = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
    dst = cv2.Canny(bn, 50, 200,apertureSize=3)
    # ah=cv2.imread(imageName,0)
    # prueba(ah)
    # cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    print(fn.shape)
    px, py, ch = fn.shape
    m = [px, py]
    threshold = int(np.amax(m) / 2)

    # lines = cv2.HoughLines(dst, 1, np.pi / 180, 75)
    lines = cv2.HoughLines(dst, 1, np.pi / 180, threshold)
    # lines = cv2.HoughLines(dst, 1, np.pi / 180, 1.5, 0.0)
    if (lines is None):
        print("posible circulo")
        graph.pruebaCirculo(src, None)
        return

    a, b, c = lines.shape
    # linesConverted = np.zeros(dtype=np.float32,shape=(len(lines),2))
    linesConverted = []
    for i in lines:
        # rho = lines[i][0][0]
        # theta = lines[i][0][1]
        rho,theta = i[0]
        a = math.cos(theta)
        b = math.sin(theta)
        x0, y0 = a * rho, b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        cv2.line(src, pt1, pt2, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow("line", src)
        cv2.destroyAllWindows()
        punto = np.array([pt1,pt2])
        linesConverted.append(punto)
        # cv2.imshow("mm",src)
        # # cv2.waitKey(2000)
        # cv2.destroyAllWindows()
    # cv2.imshow("mm",src)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    Vect.acumularPuntosInterseccion(np.asarray(linesConverted), src)
    cv2.imshow("source", fn)
    # cv2.imshow("detected lines", cdst)
    cv2.waitKey()
    cv2.destroyAllWindows()






def usoHOG(img=None):
    if(img==None):
        img = cv2.imread("./test2.jpg",0)
    img2=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imres=cv2.resize(img,(64,128))
    # nbins = 9
    # winSize = (32, 16)
    # blockSize = (8, 8)
    # blockStride = (8, 8)
    # cellSize = (8, 8)
    winSize = (128, 64)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    derivAperture = 0
    winSigma = -1
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 0
    nlevels = 64
    hog2=cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
    locat=[]

    salida=hog2.compute(img,(0,0),(0,0),locat)
    ##llamada para obtener imagen

    blocks = (winSize[1]/(blockStride[1])+winSigma)
    cellsBlock=(winSize[0]/(blockStride[0])+winSigma )
    length = blocks * cellsBlock *(nbins)*4
    print (hog2.getDescriptorSize())
    print ("prueba de HOG")
    cv2.imshow("imagen",imres)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    hog = cv2.HOGDescriptor()
    #hog.setSVMDetector(cv2.HOGDESCRIPTOR_DEFAULT_NLEVELS)
    des=[]
    lst=[]


    hist=hog.compute(img2,(0,0),(0,0),lst)
    ow,oh,ch=img.shape

    print (hist.shape)
    hist2 = hog.computeGradient(img)

    # cv2.imshow("algo",hist)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    #
    # print (hist)


def otrosEjemplos():
    img = cv2.imread("./rect2.png")
    squares = graph.find_squares(img)
    cv2.drawContours(img, squares, -1, (0, 255, 0), 3)
    cv2.imshow('squares', img)
    cv2.waitKey()
    cv2.destroyAllWindows()
# otrosEjemplos()
# usoHOG()

train()
# pruebaLineas(None,None)
# lineas("./ceda.jpg")
# lineas("./triangulo.jpg")
# lineas("./rectanguloS.jpg")
# lineas("./ceda.jpg")
# lineas("./paso.jpg")
# lineas("./velocidad.jpg")
# print (seg_intersect( p1,p2, p3,p4))
# pruebaCirculo("./ceda.jpg","ceda")