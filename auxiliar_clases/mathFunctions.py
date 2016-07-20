import math

import cv2
import numpy as np

from auxiliar_clases import graphic_features as graph


# Funciones auxiliares a mandar a un archivo
# Estas funciones son solo de puntos y calculos matematicos

IMAGEMASKREL = 20

def coseno(numero):
    return math.cos(numero)


def seno(numero):
    return math.sin(numero)


def tangente(numero):
    return math.tan(numero)


def perp( a ) :
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

# line segment a given by endpoints a1, a2
# line segment b given by endpoints b1, b2
# return
def seg_intersect(a1,a2, b1,b2) :
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot(dap, db)
    num = np.dot(dap, dp)
    output = (num / denom.astype(float)) * db + b1
    return output
    # return int(math.ceil((num / denom.astype(float))*db + b1))


def det(a, b):
    return a[0] * b[1] - a[1] * b[0]


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])  # Typo was here

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def estaPuntoenLista(list, ndarray):
    for p in list:
        if (p[0] == ndarray[0] and p[1] == ndarray[1]):
            return True
    return False


def angle_cos(p0, p1, p2):
    d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))


def LimpiarPuntosParecidos(lst, ranMin, ranMax, scale):
    ranMin = ranMin + (10 * scale)
    ranMax = ranMax + (10 * scale)
    puntoParecido = []
    for punto in lst:
        if (EsPuntoParecido(lst, punto, ranMin, ranMax)):
            puntoParecido.append(punto)
            lst.remove(punto)

    return puntoParecido


def EsPuntoParecido(lst, pt, ranMin, ranMax):
    # print(colored(pt,'red'))
    puntoParecido = []
    for lastPoint in lst:
        # lastPoint = lst[index]
        if (lastPoint[0] == 0 and lastPoint[1] == 0):
            return False
        if (lastPoint[0] == pt[0] and lastPoint[1] == pt[1]):
            return False
        print("punto a comparar con la lista: ", pt, " --- punto de la lista : ", lastPoint)
        YminRan = lastPoint[0] - ranMin
        XminRan = lastPoint[1] - ranMin
        YmaxRan = lastPoint[0] + ranMax
        XmaxRan = lastPoint[1] + ranMax
        enRangoY = (YminRan <= pt[0] and YmaxRan >= pt[0])
        enRangoX = (XminRan <= pt[1] and XmaxRan >= pt[1])
        if (enRangoY and enRangoX):
            # esta dentro del rango por lo que no se añade
            return True
    return False


def limpiarPuntosDobles(lista):
    b_set = set(tuple(x) for x in lista)
    b = [list(x) for x in b_set]
    return b


def contienePunto(puntos, punto):
    if (len(puntos) > 0):
        r = puntos[-1]
        x, y, w, h = punto
        xR, yR, wR, hR = r
        if (x == xR and y == yR and w == wR and h == hR):
            return True
    return False


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

def getMaxCircle(list):
    max = -1
    circleList = list[0]
    X, Y, R = zip(*circleList)
    maxRad = np.amax(R)
    index = np.where(R == maxRad)[0][0]
    circle = [X[index], Y[index], R[index]]
    return circle


def getPointFromList(listIndex, listElements):
    list = []
    for i in listIndex:
        list.append(listElements[i])
    return list

def obtenerPuntosAreaExterna(im, list):
    pt = list[0]
    # cv2.circle(im,(pt[0],pt[1]),2,(255,0,0),-1)
    test = im.copy()
    x, y, ch = im.shape
    W, H = (x / IMAGEMASKREL), (y / IMAGEMASKREL)
    maskShapeX = int(np.rint(W + 0.1))
    maskShapeY = int(np.rint(H + 0.1))
    list.sort(key=lambda tup: tup[1])
    mini = list[0]
    maxi = list[-1]
    maxValueY = maxi[1]
    minValueY = mini[1]
    # graph.drawPoints(im.copy(),list)
    # cv2.circle(im,(mini[0],mini[1]),4,(0,0,255),-1)
    # cv2.imshow("ha",im)

    ########################################

    mask = np.zeros((maskShapeX, maskShapeY), np.uint8)
    for pt in list:
        votofinal = pt
        votoNet = (int(votofinal[0] // IMAGEMASKREL), int(votofinal[1] // IMAGEMASKREL))  # coordenada (x,y)
        if (votoNet[0] >= 0 and votoNet[1] >= 0):
            if (votoNet[0] < maskShapeX and votoNet[1] < maskShapeY):
                mask[votoNet[1]][votoNet[0]] = 255
    mask2 = cv2.resize(mask, (y, x), None, 0, 0, cv2.INTER_NEAREST)
    cv2.imshow("mascara", mask)
    cv2.imshow("mask2", mask2)
    cv2.waitKey()
    cv2.destroyAllWindows()
    puntosMascara = np.where(mask2 == 255)
    LX, LY = puntosMascara[1], puntosMascara[0]
    # mask2[LX[0],LY[-1]]=255
    maxY = np.amax(LY)
    minY = np.amin(LY)
    YMinListIndex = np.where(LY == minY)[0]
    YMaxListIndex = np.where(LY == maxY)[0]
    listX = getPointFromList(YMinListIndex, LX)
    minX = np.amin(listX)
    maxX = np.amax(listX)
    ptMin = (minX, minY)
    ptMinMax = (maxX, minY)
    listX = getPointFromList(YMaxListIndex, LX)
    maxX = np.amax(listX)
    minX = np.amin(listX)
    ptMax = (maxX, maxY)
    ptMaxMin = (minX, maxY)
    # minX = LX[0]
    # maxX = LX[-1]

    # ptMaxMin = (maxX,minY)#MAL
    # ptMinMax = (minX,maxY)#MAL
    shapeListPoints = [ptMax, ptMin, ptMinMax, ptMaxMin]
    cv2.circle(test, ptMax, 5, (255, 0, 0), -1)
    cv2.circle(test, ptMin, 5, (255, 0, 0), -1)
    cv2.circle(test, ptMaxMin, 5, (0, 255, 0), -1)
    cv2.circle(test, ptMinMax, 5, (0, 0, 255), -1)
    cv2.imshow("img", test)
    cv2.imshow("mask2", mask2)
    cv2.waitKey()
    cv2.destroyAllWindows()

    listaPuntosFinal = limpiarPuntosDobles(shapeListPoints)

    # graph.drawPoints(im, listaPuntosFinal)
    PuntosParecidos = LimpiarPuntosParecidos(listaPuntosFinal, 5, 5, 5)

    pts = len(listaPuntosFinal)
    return listaPuntosFinal
    ########################################
    a, b = zip(*list)
    puntosEjeX, puntosEjeY = np.asarray(a), np.asarray(b)
    listaIndicesYMin = np.where((puntosEjeY > minValueY - 10) & (puntosEjeY <= minValueY + 10))[0]
    listaIndicesYMax = np.where((puntosEjeY > maxValueY - 10) & (puntosEjeY <= maxValueY + 10))[0]
    yMax = np.amax(puntosEjeY)
    indice = np.where(puntosEjeY == yMax)[0]
    l = []
    for i in indice:
        l.append([puntosEjeX[i], puntosEjeY[i]])
    graph.drawPoints(test,l)
    listaValores = []
    listaValoresMaximos = []
    for index in listaIndicesYMax:
        listaValoresMaximos.append(list[index])
    maximo = map(max, zip(*listaValoresMaximos))
    x1, x2 = maximo
    maximo = [x1, x2]
    #graph.getAndDrawPoints(im, listaIndicesYMax, list,False)
    for index in listaIndicesYMin:
        listaValores.append(list[index])
    minimo = map(min, zip(*listaValores))
    x1, x2 = minimo
    minimo = [x1, x2]
    # graph.getAndDrawPoints(im, listaIndicesYMin, list,False)
    listas = [minimo, maximo]
    #graph.drawPoints(test,listas.copy())
    MaxYminX = map(min, zip(*listaValoresMaximos))
    x1, x2 = MaxYminX
    MaxYminX=[x1,x2]
    listas.append(MaxYminX)
    MinYMaxX = map(max, zip(*listaValores))
    x1, x2 = MinYMaxX
    MinYMaxX = [x1,x2]
    listas.append(MinYMaxX)
    listaPuntos = []
    #graph.drawPoints(test,listas.copy())
    for punto in listas:
        x, y = punto
        listaPuntos.append([x, y])

    listaPuntosFinal = limpiarPuntosDobles(listaPuntos)
    graph.drawPoints(im, listaPuntosFinal)
    PuntosParecidos = LimpiarPuntosParecidos(listaPuntosFinal, 5, 5,5)

    pts = len(listaPuntosFinal)
    return listaPuntosFinal


def maximizarPuntos(im, list):
    # >> > from operator import itemgetter
    # >> > L = [[0, 1, 'f'], [4, 2, 't'], [9, 4, 'afsd']]
    # >> > sorted(L, key=itemgetter(2))
    # [[9, 4, 'afsd'], [0, 1, 'f'], [4, 2, 't']]
    print("------------------------------------------------------")
    print(list)
    ima = im.copy()

    listaPuntos = obtenerPuntosAreaExterna(im, list)
    return listaPuntos


def acumularPuntosInterseccion(lines, im):
    # cv2.imshow("mm",im)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    im2 = im.copy()
    cv2.imshow("image", im)
    cv2.waitKey(200)
    cv2.destroyAllWindows()
    imShape = im.shape
    linesShape = lines.shape
    ruptura = []
    sizeLines = len(lines)
    for i in range(sizeLines):
        for j in range(i + 1, sizeLines):
            linea1 = lines[i]
            linea2 = lines[j]
            x00, y00 = linea1[0]
            x01, y01 = linea1[1]
            x10, y10 = linea2[0]
            x11, y11 = linea2[1]
            # cv2.line(im2, (int(x00), int(y00)), (int(x01), int(y01)), (255, 0, 0), 1, cv2.LINE_AA)
            # cv2.line(im2, (int(x10), int(y10)), (int(x11), int(y11)), (255, 0, 0), 1, cv2.LINE_AA)
            # ##MIRAR si las lineas son paralelas
            point = seg_intersect(lines[i][0], lines[i][1], lines[j][0], lines[j][1])
            # if(point[0]>0):
            #     cv2.circle(im2, (int(point[0]), int(point[1])), 4, (0, 255, 0), -1)
            #     cv2.imshow("mm", im2)
            #     cv2.destroyAllWindows()
            if (point[0] >= 0 and point[1] >= 0 and point[1] < imShape[0] and point[0] < imShape[1]):
                if (not estaPuntoenLista(ruptura, point)):
                    puntoEntero = np.rint(point)
                    puntoEntero = puntoEntero.astype(int, copy=True)
                    ruptura.append(puntoEntero)
    ruptura = maximizarPuntos(im, ruptura)
    return ruptura
