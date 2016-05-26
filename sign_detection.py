# Author: Jose Miguel Buenaposada 2015.
# Simple MSER application for traffic sign window proposal generation
import math
import os
from operator import itemgetter

import cv2
import numpy as np
from matplotlib import pyplot as plt
from termcolor import colored

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

def obtenerAreas(imagen, delta, minArea):
    listaAreas = []

    return listaAreas
def getHOGDescriptorVisualImage():
    print("parsing")
    ##

    ##


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
                print("Test, processing ", filename, "\n")
                full_path = os.path.join(parcial_path, filename)
                regions = []
                I = cv2.imread(full_path)
                I = cv2.imread("./auxiliar_images/ceda2.jpg")
                color = imitacionHOG(I)
                colour = ""
                if (color == 0):  # azul
                    colour = "blue"
                elif (color == 2):  # rojo
                    colour = "red"
                lineas(I)
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


                    # print ("W/H ==> ",(w/h))
                    # hay que hacer un filtro como el de aspect ratio pero con la aparencia de los puntos(x,y,w,h)
                    if (float(w) / float(h) > 1.2) or (float(w) / float(h) < 0.8):
                        continue

                    if (not Vect.contienePunto(rects, rect)):

                        # print (w*h)
                        # cv2.rectangle(Icopy, (x, y), (x + w, y + h), (0, 255, 0), thickness=1)
                        # cv2.imshow("area", Icopy)
                        # cv2.waitKey()
                        # cv2.destroyWindow("area")
                        # Icopy = Icopy2.copy()


                        xS, yS, wS, hS = x, y, w, h
                        nuevaImagen = I[xS:xS + wS, yS:yS + hS]

                        ## pasar HOG a la nuevaImagen y si pasa el filtro de los colores, buscar lineas/circulos
                        cv2.imshow("original", I)
                        cv2.imshow("nuevaImagen", nuevaImagen)
                        # cv2.waitKey()
                        cv2.destroyAllWindows()
                        # pruebaCirculo(nuevaImagen,filename)
                        # xS,yS,wS,hS=rect
                        color = imitacionHOG(nuevaImagen)
                        colour = ""
                        if (color == 0):  # azul
                            colour = "blue"
                        elif (color == 2):  # rojo
                            colour = "red"
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

    lines = cv2.HoughLines(dst,1,np.pi/180,100)
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

def limpiarImagen(bgr_img,shapeName):
    if(shapeName=="circulo"):
        if bgr_img.shape[-1] == 3:  # color image
            b, g, r = cv2.split(bgr_img)  # get b,g,r
            rgb_img = cv2.merge([r, g, b])  # switch it to rgb
            gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = bgr_img
        # img = cv2.medianBlur(gray_img, 5)
        img = cv2.GaussianBlur(gray_img, (9, 9), 2, sigmaY=2)
        cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return (rgb_img.copy(),img)
    else:
        if bgr_img.shape[-1] == 3:  # color image
            b, g, r = cv2.split(bgr_img)  # get b,g,r
            rgb_img = cv2.merge([r, g, b])  # switch it to rgb
            gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = bgr_img
        # img = cv2.medianBlur(gray_img, 5)
        img = cv2.GaussianBlur(gray_img, (9, 9), 2, sigmaY=2)
        img= cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,75,10)
        img = cv2.bitwise_not(img)
        return (rgb_img.copy(),img)

def pruebaLineas(im,imageName):
    if(im is None):
        bgr_img = cv2.imread("./rectanguloConFlecha.jpg")
    else:
        bgr_img = im.copy()
    gray = cv2.cvtColor(bgr_img,cv2.COLOR_BGR2GRAY)
    cv2.imshow("mm",gray)
    cv2.waitKey()
    cv2.destroyAllWindows()

    imagen,img = limpiarImagen(bgr_img,"otro")
    rgb_img = imagen.copy()
    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    lineas = cv2.HoughLinesP(img,1, 3.14/180, 80, 100, 10)
    whOld=1000
    lineaAnterior = [0,0,0,0]
    if (lineas is not None):
        for linea in lineas:
            x1,y1,x2,y2 = linea[0]
            pt1 = (x1,y1)
            pt2 = (x2,y2)

            if(1):
                print ("linea: punto1 =  ",pt1," punto 2 =",pt2)

                cv2.line(imagen,(x1,y1),(x2,y2),(0,0,255),2)
                cv2.circle(imagen, (x1, y1), 3, (255, 0, 0), -1)
                cv2.circle(imagen, (x2, y2), 3, (255, 0, 0), -1)
                cv2.imshow("mm",imagen)
                cv2.waitKey(5000)
                cv2.destroyAllWindows()
                whOld=x2/y2
                imagen = rgb_img.copy()
        # plt.figure(imageName)
        # plt.subplot(121), plt.imshow(rgb_img)
        # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        # plt.subplot(122), plt.imshow(imagen)
        # plt.title('Hough Transform'), plt.xticks([]), plt.yticks([])
        # plt.show()
    else:
        print("HOLA")



def pruebaCirculo(im,imageName):
    if(im is None):
        bgr_img = cv2.imread('./test2.jpg')  # read as it is
    elif(isinstance(im,str)):
        bgr_img = cv2.imread(im)
    else:
        bgr_img = im.copy()

       # bgr_img = cv2.imread("C:/Users/xancr/Documents/Imgs/Final_Training/Images/00000/00000_00000.ppm")
    # cv2.imshow("im",bgr_img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # if bgr_img.shape[-1] == 3:  # color image
    #     b, g, r = cv2.split(bgr_img)  # get b,g,r
    #     rgb_img = cv2.merge([r, g, b])  # switch it to rgb
    #     gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    # else:
    #     gray_img = bgr_img
    # imagen = rgb_img.copy()
    # # img = cv2.medianBlur(gray_img, 5)
    # img = cv2.GaussianBlur(gray_img,(9,9),2,sigmaY=2)
    imagen,img = limpiarImagen(bgr_img,"circulo")
    rgb_img = imagen.copy()
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 2, 100,
                               param1=10, param2=10, minRadius=0, maxRadius=0)
    if (circles is not None):
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(imagen, (i[0], i[1]), i[2], (0, 255, 0), 1)
            # draw the center of the circle
            cv2.circle(imagen, (i[0], i[1]), 1, (0, 0, 255), 2)

        # cv2.imshow("im",imagen)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        plt.figure(imageName)
        plt.subplot(121), plt.imshow(rgb_img)
        plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(imagen)
        plt.title('Hough Transform'), plt.xticks([]), plt.yticks([])
        plt.show()

    else:
        print (colored("NO hay circulos",'red'))

def detectarCirculo(imagen):
    print("detectar circulo")
    #image = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)
    imBN=cv2.imread("./test2.jpg",cv2.IMREAD_GRAYSCALE)
    (thresh, im_bw) = cv2.threshold(imBN, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # cv2.imshow("bnn",im_bw)
    # cv2.imshow("bn",imBN)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    circles = cv2.HoughCircles(im_bw,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=0)
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        #draw the outer circle
        cv2.circle(imBN,(i[0],i[1]),i[2],(0,255,0),2)
        #draw the center of the circle
        cv2.circle(imBN,(i[0],i[1]),2,(0,0,255),3)
    cv2.imshow('detected circles',imBN)
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


def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def find_squares(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    squares = []
    for gray in cv2.split(img):
        for thrs in range(0, 255, 26):
            if thrs == 0:
                bin = cv2.Canny(gray, 0, 50, apertureSize=5)
                bin = cv2.dilate(bin, None)
            else:
                retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
            bin, contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
                if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in range(4)])
                    if max_cos < 0.1:
                        squares.append(cnt)
    return squares


def obtenerPuntosAreaExterna(im,list):
    listaPuntos=[]
    a,b = zip(*list)
    maxY=(map(max, zip(*list)))
    minX=map(min, zip(*list))
    x1,y1 = maxY
    x2,y2 = minX
    cv2.circle(im, (int(x1), int(y1)), 4, (0, 0, 255), -1)
    cv2.circle(im, (int(x2), int(y2)), 4, (0, 255, 0), -1)
    cv2.imshow("mmm", im)
    cv2.destroyAllWindows()
    puntosEjeX,puntosEjeY = np.asarray(a),np.asarray(b)
    ptoMin = np.where(puntosEjeX==x2)
    indicesY = []
    for indice in ptoMin[0]:
        indicesY.append(puntosEjeY[indice])
    minMaxY = np.amax(indicesY)
    minMax = [x2,minMaxY]
    ptoMax = np.where(puntosEjeX==x1)
    indicesY = []
    for indice in ptoMax[0]:
        indicesY.append(puntosEjeY[indice])
    minY = np.amin(indicesY)
    maxMin= [x1,minY]
    listaPuntos =[[x1,y1],[x2,y2],minMax,maxMin]
    cv2.circle(im,(maxMin[0],maxMin[1]),4,(255,0,0),-1)

    cv2.circle(im,(minMax[0],minMax[1]),4,(255,0,0),-1)

    # cv2.line(im,pt1=(int(x),int(y)),pt2=(int(z),int(w)),color=(255,0,0),thickness=1,lineType=cv2.LINE_AA)
    cv2.imshow("mmm",im)
    cv2.destroyAllWindows()
    listaPuntosFinal = Vect.limpiarPuntosDobles(listaPuntos)
    pts = len(listaPuntosFinal)
    if (pts < 3):
        print("circle")
    elif (pts == 3):
        print("triangle")
    elif (pts == 4):
        print("square")
    else:
        print("other shape")
    # print ("el punto en el eje X minimo es :",z," el punto encontrado es :",puntosEjeX[ptoMin[0][0]])
    # gggg = ara[(ara <(x+2))&(ara >(x-2))]
    # print (colored("el maximo es: ",'red'),(x,y))
    # print (colored("el minimo es: ",'green'),(z,w))
    return listaPuntos

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
    cv2.destroyAllWindows()

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
        if (not Vect.puntoParecido(shape, pt, 20, 20, index)):
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

    cv2.destroyAllWindows()
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
    squares = find_squares(img)
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