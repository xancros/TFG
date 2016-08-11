import math

import cv2
import numpy as np

# Funciones auxiliares a mandar a un archivo
# Estas funciones son solo de puntos y calculos matematicos

IMAGEMASKREL = 20  # 10

def coseno(numero):
    return math.cos(numero)


def seno(numero):
    return math.sin(numero)


def tangente(numero):
    return math.tan(numero)


def getRects(listOfPoints):
    subList = listOfPoints.copy()
    vects = []
    for i in range(0, 3):
        source = subList[i]
        rects = []
        for j in range(0, 3):
            end = subList[j]
            # if (source == end):
            if (np.array_equal(source, end)):
                continue

            pt0 = end[0] - source[0]
            pt1 = end[1] - source[1]
            rect = [pt0, pt1]
            rects.append(rect)
        vects.append(rects)
    return vects


def checkAngle(listOfPoints):
    # print("Checking angle value from list")
    list = np.asarray(listOfPoints, np.int32)
    vects = getRects(list)
    angles = []
    angleMin = np.float64(50)
    angleMax = np.float64(70)
    for par in vects:
        vect1, vect2 = par
        angle = getAngleVectors(vect1, vect2)
        angles.append(angle)
        # if (not (angle > 45 and angle <= 70)):
        #     return False
    nIndex = np.where(((angles > angleMin) & (angle <= angleMax)))[0]
    if (len(nIndex) >= 2):
        return True
    else:
        return False


def getAngleVectors(V1, V2):
    # print("Getting angle of vectors...")
    s1 = np.dot(V1[0], V2[0])
    s2 = np.dot(V1[1], V2[1])
    sum = s1 + s2
    m1 = getModule(V1)
    m2 = getModule(V2)
    dot = np.dot(m1, m2)
    cos = sum / dot
    angle = np.arccos(cos)
    angle2 = np.rad2deg(angle)
    return angle2


def getModule(Vector):
    vector2 = np.power(Vector, 2)
    sum = vector2[0] + vector2[1]
    sqr = np.sqrt(sum)
    return sqr

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
    op = 0
    if (denom.astype(float) == 0):
        op = 0
    else:
        op = (num / denom.astype(float))
    output = op * db + b1
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



def LimpiarPuntosParecidos(lst, ranMin, ranMax, scale):
    # ranMin = ranMin + (10 * scale)
    # ranMax = ranMax + (10 * scale)
    ranMin = ranMin + (5 * scale)
    ranMax = ranMax + (5 * scale)
    puntoParecido = []
    listCopy = lst.copy()
    lenght = len(lst)
    for index in range(0, lenght):
        punto = listCopy[index]
        if (EsPuntoParecido(lst, punto, ranMin, ranMax)):
            lst.remove(punto)
    return puntoParecido


def EsPuntoParecido(lst, pt, ranMin, ranMax):
    # print(colored(pt,'red'))
    puntoParecido = []
    for lastPoint in lst:
        # print("punto a comparar con la lista: ", pt, " --- punto de la lista : ", lastPoint)
        # lastPoint = lst[index]
        if (lastPoint[0] == 0 and lastPoint[1] == 0):
            continue
        if (lastPoint[0] == pt[0] and lastPoint[1] == pt[1]):
            continue
        # print("punto a comparar con la lista: ", pt, " --- punto de la lista : ", lastPoint)
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


def limpiarListaDobles(lista):
    b_set = set(x for x in lista)
    b = [x for x in b_set]
    return b

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
    if (len(list) == 0):
        return []
    pt = list[0]
    # graph.drawPoints(im.copy(),list)
    # cv2.circle(im,(pt[0],pt[1]),2,(255,0,0),-1)
    test = im.copy()
    x, y, ch = im.shape
    scale = x / 10
    W, H = (x / IMAGEMASKREL), (y / IMAGEMASKREL)
    maskShapeX = int(np.rint(W + 0.1))
    maskShapeY = int(np.rint(H + 0.1))
    #list.sort(key=lambda tup: tup[1])
    mini = list[0]
    maxi = list[-1]
    maxValueY = maxi[1]
    minValueY = mini[1]
    # graph.drawPoints(im.copy(),list)
    # cv2.circle(im,(mini[0],mini[1]),4,(0,0,255),-1)
    # cv2.imshow("ha",im)

    ########################################

    mask = np.zeros((maskShapeX, maskShapeY), np.uint8)
    # mask = np.zeros((x, y), np.uint8)
    for pt in list:
        votofinal = pt
        votoNet = (int(votofinal[0] // IMAGEMASKREL), int(votofinal[1] // IMAGEMASKREL))  # coordenada (x,y)
        if (votoNet[0] >= 0 and votoNet[1] >= 0):
            if (votoNet[0] < maskShapeX and votoNet[1] < maskShapeY):
                mask[votoNet[1]][votoNet[0]] = 255
    mask2 = cv2.resize(mask, (y, x), None, 0, 0, cv2.INTER_NEAREST)
    # cv2.imshow("mascara", mask)
    # cv2.imshow("mask2", mask2)
    # cv2.waitKey(1500)
    # cv2.destroyWindow("mascara")
    # cv2.destroyWindow("mask2")
    puntosMascara = np.where(mask2 == 255)
    LX, LY = puntosMascara[1], puntosMascara[0]
    #########################
    rangeValue = 30
    listOrdered = sorted(list, key=lambda point: point[1])
    yMin = listOrdered[0][1]
    yMax = listOrdered[-1][1]
    ranYMin = int(yMin - rangeValue)
    ranYMinMax = int(yMin + rangeValue)
    if (ranYMin < 0): ranYMin = 0
    ranYMax = int(yMax + rangeValue)
    if (ranYMax > y): ranYMax = y
    ranYMaxMin = int(yMax - rangeValue)
    listaX, listaY = zip(*listOrdered.copy())
    listaX = np.asarray(listaX, np.uint)
    listaY = np.asarray(listaY, np.uint)
    indexYMinRange = np.where(((listaY > ranYMin) & (listaY < ranYMinMax)))
    listXMin = []
    for index in indexYMinRange:
        listXMin.append(listaX[index])
    if (len(listXMin) == 1):
        elem = listXMin[0]
        if (elem.size == 0):
            return []
    minYminX = (np.amin(listXMin), yMin)
    minYmaxX = (np.amax(listXMin), yMin)
    # np.where( ((listaY>ranYMin) & (listaY<ranYMinMax)) )
    indexYMaxRange = np.where(((listaY > ranYMaxMin) & (listaY < ranYMax)))

    listXMax = []
    for index in indexYMaxRange:
        listXMax.append(listaX[index])
    if (len(listXMax) == 1):
        elem = listXMax[0]
        if (elem.size == 0):
            return []
    maxYminX = (np.amin(listXMax), yMax)
    maxYmaxX = (np.amax(listXMax), yMax)
    ll = [minYminX, minYmaxX, maxYminX, maxYmaxX]
    # graph.drawPoints(test.copy(), ll.copy())
    ll = limpiarPuntosDobles(ll)
    puntosParecidos = LimpiarPuntosParecidos(ll, 5, 5, scale)
    return ll
    print(indexYMinRange)

    #############
    #
    #   -----------------------------------------
    #
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
    cv2.waitKey(1500)
    cv2.destroyWindow("img")
    cv2.destroyWindow("mask2")
    test = im.copy()
    listaPuntosFinal = limpiarPuntosDobles(shapeListPoints)

    # graph.drawPoints(im, listaPuntosFinal)
    PuntosParecidos = LimpiarPuntosParecidos(listaPuntosFinal, 5, 5, scale)
    # graph.drawPoints(test,listaPuntosFinal)
    pts = len(listaPuntosFinal)
    ########################################

    return listaPuntosFinal


def maximizarPuntos(im, list):
    # >> > from operator import itemgetter
    # >> > L = [[0, 1, 'f'], [4, 2, 't'], [9, 4, 'afsd']]
    # >> > sorted(L, key=itemgetter(2))
    # [[9, 4, 'afsd'], [0, 1, 'f'], [4, 2, 't']]
    # print("------------------------------------------------------")
    # print(list)
    ima = im.copy()

    listaPuntos = obtenerPuntosAreaExterna(im, list)
    return listaPuntos


def acumularPuntosInterseccion(lines, im):
    im2 = im.copy()
    imShape = im.shape

    ruptura = []
    scale = imShape[0]/10
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
            if (point[0] >= 0 and point[1] >= 0 and point[1] < imShape[0] and point[0] < imShape[1]):
                if (not estaPuntoenLista(ruptura, point)):
                    puntoEntero = np.rint(point)
                    puntoEntero = puntoEntero.astype(int, copy=True)
                    ruptura.append(puntoEntero)
                    puntos = ruptura.copy()
                    ruptura = limpiarPuntosDobles(ruptura)
                    # graph.drawPoints(im2.copy(), ruptura)
                    # PuntosParecidos = LimpiarPuntosParecidos(ruptura, 5, 5, 10)
    # graph.drawPoints(im2.copy(), ruptura)
    ruptura = maximizarPuntos(im, ruptura)
    return ruptura
