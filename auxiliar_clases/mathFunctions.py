import numpy as np


# Funciones auxiliares a mandar a un archivo
# Estas funciones son solo de puntos y calculos matematicos
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


def LimpiarPuntosParecidos(lst, ranMin, ranMax):
    puntoParecido = []
    for punto in lst:
        if (EsPuntoParecido(lst, punto, ranMin, ranMax, 0)):
            puntoParecido.append(punto)
            lst.remove(punto)

    return puntoParecido


def EsPuntoParecido(lst, pt, ranMin, ranMax, index):
    # print(colored(pt,'red'))
    puntoParecido = []
    for lastPoint in lst:
        # lastPoint = lst[index]
        if (lastPoint[0] == 0 and lastPoint[1] == 0):
            return False
        if (lastPoint[0] == pt[0] and lastPoint[1] == pt[1]):
            continue
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
    #####################
