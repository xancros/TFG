# Author: Jose Miguel Buenaposada 2015.
# Simple MSER application for traffic sign window proposal generation
import math
import os
from operator import itemgetter

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
                # I = cv2.imread("./auxiliar_images/ceda.jpg")
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
                    A = x
                    B = x+w
                    C = y
                    D = y+h
                    if (float(w) / float(h) > 1.2) or (float(w) / float(h) < 0.8):
                        continue

                    if (not Vect.contienePunto(rects, rect)):
                        cv2.rectangle(Icopy, (x, y), (x + w, y + h), (0, 255, 0), thickness=1)
                        cv2.imshow("area", Icopy)
                        cv2.waitKey()
                        cv2.destroyWindow("area")


                        xS, yS, wS, hS = x, y, w, h
                        nuevaImagen = I[xS:xS + wS, yS:yS + hS]

                        ## pasar HOG a la nuevaImagen y si pasa el filtro de los colores, buscar lineas/circulos
                        cv2.imshow("original", I)
                        cv2.imshow("nuevaImagen", nuevaImagen)
                        cv2.destroyAllWindows()
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
    lines = cv2.HoughLines(dst, 1, np.pi / 180, 75)
    # lines = cv2.HoughLines(dst, 1, np.pi / 180, 1.5, 0.0)
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

    acumularPuntosInterseccion(np.asarray(linesConverted),fn)
    cv2.imshow("source", fn)
    # cv2.imshow("detected lines", cdst)
    cv2.waitKey()






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


def obtenerPuntoParecido(ListaPuntos, punto, Maximo=True):
    a, b = zip(*ListaPuntos)
    puntosEjeX, puntosEjeY = np.asarray(a), np.asarray(b)
    ptX, ptY = punto
    listaIndicesX, listaPuntosY = []
    minX = punto[0] - 3
    maxX = punto[0] + 3
    index = 0
    puntoObtenido = ()
    for x in puntosEjeX:
        if (x >= minX and x <= maxX):
            listaIndicesX.append(index)
        index += 1
    for i in index:
        listaPuntosY.append(puntosEjeY[i])

    if (Maximo):
        maximoY = np.amax(listaPuntosY)
        indices = np.where(listaPuntosY == maximoY)[0]

    else:
        minimoY = np.amin(listaPuntosY)
        indices = np.where(listaPuntosY == minimoY)
    puntoObtenido = (punto[0], listaPuntosY[indices[0]])
    return puntoObtenido


##

def obtenerPuntosAreaExterna(im,list):
    # probar a hacer una búsqueda por altura en los puntos y despues buscar maximo y minimo de X en rango de +-3 o asi
    pt = list[0]
    # cv2.circle(im,(pt[0],pt[1]),2,(255,0,0),-1)

    list.sort(key=lambda tup: tup[1])
    mini = list[0]
    maxi = list[-1]
    maxValueY = maxi[1]
    minValueY = mini[1]
    # cv2.circle(im,(mini[0],mini[1]),4,(0,0,255),-1)
    # cv2.imshow("ha",im)
    a, b = zip(*list)
    puntosEjeX, puntosEjeY = np.asarray(a), np.asarray(b)
    listaIndicesYMin = np.where((puntosEjeY > minValueY - 5) & (puntosEjeY <= minValueY + 5))[0]
    listaIndicesYMax = np.where((puntosEjeY > maxValueY - 5) & (puntosEjeY <= maxValueY + 5))[0]
    listaValores = []
    listaValoresMaximos = []
    for index in listaIndicesYMax:
        listaValoresMaximos.append(list[index])
    maximo = map(max, zip(*listaValoresMaximos))
    graph.getAndDrawPoints(im, listaIndicesYMax, list)
    for index in listaIndicesYMin:
        listaValores.append(list[index])
    minimo = map(min, zip(*listaValores))
    graph.getAndDrawPoints(im, listaIndicesYMin, list)
    MinXY = minimo
    maximoXY = maximo
    listas = [MinXY, maximoXY]
    MaxYminX = map(min, zip(*listaValoresMaximos))
    listas.append(MaxYminX)
    MinYMaxX = map(max, zip(*listaValores))
    listas.append(MinYMaxX)
    listaPuntos = []
    for punto in listas:
        x, y = punto
        listaPuntos.append([x, y])

    listaPuntosFinal = Vect.limpiarPuntosDobles(listaPuntos)
    PuntosParecidos = Vect.LimpiarPuntosParecidos(listaPuntosFinal, 20, 20)

    pts = len(listaPuntosFinal)
    if (pts < 3):
        print("circle")
    elif (pts == 3):
        print("triangle")
    elif (pts == 4):
        print("square")
    else:
        print("other shape")
    return listas

def maximizarPuntos(im,list):
    # >> > from operator import itemgetter
    # >> > L = [[0, 1, 'f'], [4, 2, 't'], [9, 4, 'afsd']]
    # >> > sorted(L, key=itemgetter(2))
    # [[9, 4, 'afsd'], [0, 1, 'f'], [4, 2, 't']]
    print("------------------------------------------------------")
    print(list)
    ima = im.copy()
    # sorted(list,key=itemgetter(0))
    for pt in list:
        cv2.circle(ima, (pt[0], pt[1]), 3, (255, 0, 0), -1)
    cv2.imshow("aaa", ima)
    #cv2.destroyAllWindows()

    listaPuntos=obtenerPuntosAreaExterna(im,list)
    list.sort(key=itemgetter(0,1), reverse=True)
    print (list)
    auxiliar = []
    shape = np.zeros((4,2),np.uint32)
    shape[0]=np.rint(list[0])
    auxiliar.append(list[0])
    im2 = im.copy()
    index = 0
    for pt in list:
        x, y = pt
        if (not Vect.EsPuntoParecido(shape, pt, 20, 20, index)):
            cv2.circle(im2, (int(x), int(y)), 4, (0, 255, 0), -1)
            cv2.imshow("mm", im2)
            cv2.destroyAllWindows()
            im2=im.copy()
            shape[index+1]=np.rint(pt)
            index+=1

    # return list
    return shape
def acumularPuntosInterseccion(lines,im):
    # cv2.imshow("mm",im)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    im2 = im.copy()
    imShape = im.shape
    linesShape = lines.shape
    ruptura = []
    sizeLines = len(lines)
    for i in range(sizeLines):
        for j in range(i+1,sizeLines):
            linea1 = lines[i]
            linea2 = lines[j]
            x00,y00 = linea1[0]
            x01, y01 = linea1[1]
            x10,y10 = linea2[0]
            x11, y11 = linea2[1]
            cv2.line(im2,(int(x00),int(y00)),(int(x01),int(y01)),(255,0,0),1,cv2.LINE_AA)
            cv2.line(im2, (int(x10), int(y10)), (int(x11), int(y11)), (255, 0, 0), 1, cv2.LINE_AA)

            point = Vect.seg_intersect(lines[i][0], lines[i][1], lines[j][0], lines[j][1])
            # if(point[0]>0):
            #     cv2.circle(im2, (int(point[0]), int(point[1])), 4, (0, 255, 0), -1)
            #     cv2.imshow("mm", im2)
            #     cv2.destroyAllWindows()
            if(point[0]>=0 and point[1] >=0 and point[1]<imShape[0] and point[0]<imShape[1]):
                if (not Vect.estaPuntoenLista(ruptura, point)):
                    puntoEntero = np.rint(point)
                    puntoEntero=puntoEntero.astype(int,copy=True)
                    ruptura.append(puntoEntero)

    cv2.imshow("mm", im2)

    #cv2.destroyAllWindows()
    im2 = im.copy()
    if(True):
        ruptura = maximizarPuntos(im,ruptura)
        print (ruptura)
        ruptura=ruptura[::-1]
        print (ruptura)
        for pt in ruptura:
            x,y = pt
            cv2.circle(im2,(int(x),int(y)),4,(0,255,0),-1)
            cv2.imshow("mm",im2)
            # cv2.waitKey()
            cv2.destroyAllWindows()


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