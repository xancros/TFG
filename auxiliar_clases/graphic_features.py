import cv2
import numpy as np

from auxiliar_clases import mathFunctions as Vect

kernel = np.ones((5, 5), np.uint8)
CH = -1
CV = -1
CS = -1
nombre = ""

# minParam2 = 13
minParam2 = 30


def printHSV_Values(image):
    imgF = image.copy()
    h, s, v = cv2.split(imgF)
    indices = np.where(imgF == 255)
    indicesX = indices[0]
    indicesY = indices[1]
    print("VALORES DE HUE (MATIZ)| VALORES DE SATURACION | VALORES DE ILUMINACION")
    zeros = np.zeros_like(imgF)
    for i in range(0, len(indicesX)):
        x = indicesX[i]
        y = indicesY[i]
        elem = h[indicesX[i]][indicesY[i]]
        ese = s[indicesX[i]][indicesY[i]]
        uve = v[indicesX[i]][indicesY[i]]
        # (elem > -1 and elem <= 10) or (elem >= 160 and elem < 180)
        if (1):
            zeros[x][y] = imgF[x][y]

            print("MATIZ: ", elem, "SATURACION: ", ese, "ILUMINACION: ", uve)

    cv2.imshow("mascara", zeros)
    cv2.imshow("hsv", imgF)
    cv2.waitKey()
    cv2.destroyAllWindows()

def redAreaDetection(image, name, show=False):
    # image2 = cv2.imread("./auxiliar_images/cirRoj.jpg")
    img = image.copy()
    sp = img.shape
    # cambiar espacio rgb -> HSV
    imag2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    x, y, ch = img.shape
    # cv2.imshow("hsvor", imag2)
    h, s, v = cv2.split(imag2)
    CH = np.zeros_like(h)
    CH = h
    CV = np.zeros_like(v)
    CV = v
    CS = np.zeros_like(s)
    CS = s
    # s += 50
    s = cv2.equalizeHist(s)
    v = cv2.equalizeHist(v)
    chs = [h, s, v]
    imgRes = cv2.merge(chs)
    test = cv2.cvtColor(imgRes, cv2.COLOR_HSV2BGR)
    # cv2.imshow("PRUEBA", test)
    # cv2.imshow("ORIGINAL", img)
    # cv2.waitKey(1000)
    # cv2.destroyAllWindows()
    hsv_1 = (110, 50, 50)
    hsv_2 = (130, 255, 255)
    # im = cv2.inRange(imag2,(0,50,50),(20,255,255))
    im = cv2.inRange(imgRes, (0, 100, 30), (10, 255, 255))
    im2 = cv2.inRange(imgRes, (160, 100, 30), (180, 255, 255))
    imgF = im + im2



    if (show):
        cv2.imshow("image", image)
        cv2.imshow("win1", im)
        cv2.imshow("win2", im2)
        cv2.imshow("win3", imgF)
        cv2.waitKey()
        cv2.destroyAllWindows()
    return imgF


def drawPoints(image, listPoints):
    for point in listPoints:
        x, y = point
        pt = (x, y)
        cv2.circle(image, pt, 5, (0, 255, 0), -1)
        cv2.imshow("Point in list", image)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
    cv2.imshow("Points in list", image)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()


def getAndDrawPoints(image, listIndex, list, ShowImage=True):
    for index in listIndex:
        pt = (list[index][0], list[index][1])
        cv2.circle(image, pt, 3, (255, 0, 0), -1)
    if (ShowImage):
        cv2.imshow("Points in list", image)
        cv2.waitKey()
        cv2.destroyAllWindows()



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
                cnt = cv2.approxPolyDP(cnt, 0.02 * cnt_len, True)
                if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([Vect.angle_cos(cnt[i], cnt[(i + 1) % 4], cnt[(i + 2) % 4]) for i in range(4)])
                    if max_cos < 0.1:
                        squares.append(cnt)
    return squares


def find_lines(image, mask):
    #blur mask of image

    img = image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst = cv2.Canny(mask.copy(), 50, 200)
    #dst = cv2.Canny(mask.copy(),50,150,apertureSize=3)
    cv2.imshow("canny", dst)
    cv2.waitKey(500)
    cv2.destroyAllWindows()
    px, py, chs = img.shape
    m = [px, py]
    # threshold = int(np.amin(m) / 2)

    threshold = int(np.amin(m) / 4)

    # lines = cv2.HoughLines(dst, 1, np.pi / 180, threshold)
    lines = cv2.HoughLines(dst, 1, np.pi / 180.0, 50, np.array([]), 0, 0)
    if ((not (lines is None)) and (len(lines) > 2)):
        a, b, c = lines.shape
        # linesConverted = np.zeros(dtype=np.float32,shape=(len(lines),2))
        linesConverted = []
        for i in lines:
            # rho = lines[i][0][0]
            # theta = lines[i][0][1]
            rho, theta = i[0]
            a = Vect.coseno(theta)
            b = Vect.seno(theta)
            x0, y0 = a * rho, b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv2.line(img, pt1, pt2, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.imshow("line", img)
            cv2.waitKey(500)
            cv2.destroyAllWindows()
            punto = np.array([pt1, pt2])
            linesConverted.append(punto)
        # cv2.imshow("mm",src)
        # # cv2.waitKey(2000)
        # cv2.destroyAllWindows()
        cv2.imshow("line", img)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        puntos = Vect.acumularPuntosInterseccion(np.asarray(linesConverted), image.copy())
        if (len(puntos) == 3):
            ## COMPROBAR QUE LAS LINEAS DE LOS PUNTOS DEL TRIANGULO SEAN AGUDOS EN TORNO A LOS 60º
            ## COMPROBAR SI HAY LINEAS PARALELAS ANTES
            ## HACER UN ACUMULADOR DE RESULTADOS ENTRE FONDO, TRIANGULOS, CIRCULOS PARA RESULTADO DE IMAGEN
            if (Vect.checkAngle(puntos)):
                return "triangle"
            return "other"
        else:
            return "other"

    return None


#
# def find_lines(image):
#     #blur mask of image
#
#     img = image.copy()
#     dst = cv2.Canny(img.copy(), 50, 200, apertureSize=3)
#     cv2.imshow("canny", dst)
#     cv2.waitKey(500)
#     cv2.destroyAllWindows()
#     px, py = img.shape
#     m = [px, py]
#     threshold = int(np.amin(m) / 2)
#     while (threshold > 0):
#         lines = cv2.HoughLines(dst, 1, np.pi / 180, threshold)
#         if (lines is None):
#             threshold -= 1
#         elif (len(lines) > 2):
#             a, b, c = lines.shape
#             # linesConverted = np.zeros(dtype=np.float32,shape=(len(lines),2))
#             linesConverted = []
#             for i in lines:
#                 # rho = lines[i][0][0]
#                 # theta = lines[i][0][1]
#                 rho, theta = i[0]
#                 a = Vect.coseno(theta)
#                 b = Vect.seno(theta)
#                 x0, y0 = a * rho, b * rho
#                 pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
#                 pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
#                 cv2.line(img, pt1, pt2, (255, 0, 0), 1, cv2.LINE_AA)
#                 cv2.imshow("line", img)
#                 cv2.waitKey(1000)
#                 cv2.destroyAllWindows()
#                 punto = np.array([pt1, pt2])
#                 linesConverted.append(punto)
#             # cv2.imshow("mm",src)
#             # # cv2.waitKey(2000)
#             # cv2.destroyAllWindows()
#             puntos = Vect.acumularPuntosInterseccion(np.asarray(linesConverted), img)
#             if (len(puntos)==3):
#                 return "triangle"
#             else:
#                 return "other"
#
#     return None

def findCircles(img, param2, minRad=0, maxRad=0):
    try:
        # circles = cv2.HoughCircles(img.copy(), cv2.HOUGH_GRADIENT, 2, 1, np.array([]), 100, param2, 1)
        circles = cv2.HoughCircles(img.copy(), cv2.HOUGH_GRADIENT, 1, 1,
                                   param1=50, param2=param2, minRadius=minRad, maxRadius=maxRad)
    except:
        circles = "nada"

    return circles


def usoHOG(img=None):
    if (img == None):
        img = cv2.imread("./test2.jpg", 0)
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imres = cv2.resize(img, (64, 128))
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
    hog2 = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                             histogramNormType, L2HysThreshold, gammaCorrection, nlevels)
    locat = []

    salida = hog2.compute(img, (0, 0), (0, 0), locat)
    ##llamada para obtener imagen

    blocks = (winSize[1] / (blockStride[1]) + winSigma)
    cellsBlock = (winSize[0] / (blockStride[0]) + winSigma)
    length = blocks * cellsBlock * (nbins) * 4
    print(hog2.getDescriptorSize())
    print("prueba de HOG")
    cv2.imshow("imagen", imres)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    hog = cv2.HOGDescriptor()
    # hog.setSVMDetector(cv2.HOGDESCRIPTOR_DEFAULT_NLEVELS)
    des = []
    lst = []

    hist = hog.compute(img2, (0, 0), (0, 0), lst)
    ow, oh, ch = img.shape

    print(hist.shape)
    hist2 = hog.computeGradient(img)

    # cv2.imshow("algo",hist)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    #
    # print (hist)


def circles(img, blurMask, cimg, ruta):
    overflow = False
    shape = img.shape
    s = [shape[0], shape[1]]
    rm = (np.amin(s) / 4)
    param2 = 100
    ccimg = cimg.copy()
    while (param2 > minParam2):
        circles = findCircles(blurMask, param2)
        if (isinstance(circles, str)):
            param2 = param2 - 1
        elif (not (circles is None)):

            a, b, c = circles.shape
            for i in range(b):
                ptX, ptY, r = circles[0][i]
                # print ((r,rm))
                if (r >= rm):

                    # print("hay circulos, paramatro2 valor = ", param2)
                    cv2.circle(ccimg, (ptX, ptY), 2, (0, 255, 0), 3)
                    cv2.circle(ccimg, (ptX, ptY), r, (255, 0, 0), 5)
                    c1 = [int(ptX - r - 3), int(ptY - r - 3)]
                    # c2=(int(ptX-r-3),int(ptY+r+3))
                    # c3=(int(ptX+r+3),int(ptY-r-3))
                    c4 = [int(ptX + r + 3), int(ptY + r + 3)]
                    if (c1[0] < -5):
                        overflow =True
                        c1[0] = 0
                    if (c1[1] < -5):
                        overflow =True
                        c1[1] = 0
                    if (c4[0] > shape[0] + 5):
                        overflow =True
                        c4[0] = shape[0]
                    if (c4[1] > shape[1] + 5):
                        overflow =True
                        c4[1] = shape[1]
                    mm = blurMask[c1[0]:c1[0] + c4[0], c1[1]:c1[1] + c4[1]]
                    cv2.rectangle(ccimg, (c1[0], c1[1]), (c4[0], c4[1]), (255, 0, 0), 3)
                    # COMPROBAR SI EL FRAGMENTO DE LA MASCARA TIENE MAS DE 500 puntos
                    x, y, ch = cimg.shape
                    # cimg = cv2.resize(cimg,(x*2,y*2))
                    # path = ruta.split("\\")
                    # print(path[-1])
                    ############################
                    if (overflow):
                        ccimg = cimg.copy()
                        overflow = False
                        continue
                    ############################
                    cv2.imshow("original", img)
                    cv2.imshow("mask", blurMask)
                    cv2.imshow("detected circles", ccimg)
                    cv2.waitKey(500)
                    cv2.destroyAllWindows()
                    # print("----> Circulo detectado en la imagen : ", path, " <------")
                    ##en un futuro hay que obtener todos los circulos detectados, ordernar y obtener el circulo
                    ## con mayor area, osea el mas grande :D
                    return "circle"


        if (param2 <= minParam2):
            # no se han detectado circulos, pasamos a detectar lineas
            # print("couldn't find important shape")
            show = cv2.resize(img.copy(), (s[0] * 2, s[1] * 2))
            path = ruta.split("\\")
            # print(path[-1])
            return None
        param2 -= 1
    return None

def shapeDetection(img, ruta):
    nombre = ((ruta.split("\\"))[-1])
    if (ruta.__contains__("\\06\\")):
        print()

    imageShape = img.shape
    Icopy = img.copy()
    Icp = Icopy.copy()
    Icblur = cv2.medianBlur(Icopy, 7)
    #Icblur = cv2.blur(Icblur, (5, 5))
    redMask = redAreaDetection(Icblur, nombre)
    mask = redMask.copy()
    shape = mask.shape
    s = [shape[0], shape[1]]
    rm = (np.amin(s) / 3)
    rmax = (np.amax(s) / 2)
    res = np.ones_like(Icopy.copy())
    thr = int(shape[0] / 8)
    blurMask = cv2.medianBlur(mask, 9)
    blurMask = cv2.blur(blurMask, (5, 5))

    ##

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(blurMask.copy(), cv2.MORPH_OPEN, kernel, iterations=2)
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(blurMask.copy(), kernel, iterations=2)
    opening = cv2.GaussianBlur(opening, (9, 9), 0)
    erosion = cv2.GaussianBlur(erosion, (9, 9), 0)
    # cv2.imshow("test",opening)
    # cv2.imshow("test2",erosion)
    # cv2.moveWindow("test2",0,0)
    # cv2.waitKey(1000)
    # cv2.destroyAllWindows()
    ##
    # blurMask = cv2.resize(blurMask, (imageShape[0] * 5, imageShape[1] * 5))
    # res = cv2.bitwise_and(Icp, Icp, mask=blurMask)
    # cv2.imshow("original", img)
    # cv2.imshow("source", Icopy)
    # cv2.imshow("bitwise", res)
    # cv2.imshow("blur", blurMask)
    # cv2.waitKey(1000)
    # cv2.destroyAllWindows()
    cimg = Icopy.copy()  # numpy function
    # circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 10, np.array([]), 100, 30, 1, 30)
    NCircles = circles(img.copy(), opening, cimg, ruta)
    if NCircles is None:
        return find_lines(Icopy.copy(), erosion.copy())
    else:
        return NCircles
