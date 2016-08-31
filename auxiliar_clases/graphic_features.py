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


def preProcessImage(image):
    channel = image.copy()
    # channel = cv2.adaptiveThreshold(channel, 255, adaptive_method=cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=55, param1=7)
    # channel = cv2.adaptiveThreshold(channel,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,55,7)
    channel = cv2.adaptiveThreshold(channel, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 55, 3)
    # mop up the dirt
    channel = cv2.dilate(channel, None, 1)
    channel = cv2.erode(channel, None, 1)
    return channel


def getBinaryInvMask(RGBImage):
    img = RGBImage.copy()
    b, g, r = cv2.split(img)
    r = preProcessImage(r)
    g = preProcessImage(g)
    b = preProcessImage(b)
    rgb = [b, g, r]
    processedImage = cv2.merge(rgb)
    # cv2.imshow("pr",processedImage)
    # cv2.waitKey(800)
    # cv2.destroyAllWindows()
    grayImage = cv2.cvtColor(processedImage, cv2.COLOR_BGR2GRAY)
    ret, threshold = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY_INV)
    return threshold


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
    cv2.destroyWindow("mascara")
    cv2.destroyWindow("hsv")

def redAreaDetection(image, name, show=False):
    img = image.copy()
    test3 = getBinaryInvMask(img)
    pp = cv2.bitwise_and(img, img, mask=test3)
    # cambiar espacio rgb -> HSV
    img = image.copy()
    b, g, r = cv2.split(img)
    r = preProcessImage(r)
    g = preProcessImage(g)
    b = preProcessImage(b)
    rgb = [b, g, r]
    prc = cv2.merge(rgb)
    prueba = cv2.cvtColor(prc, cv2.COLOR_BGR2HSV)
    imag2 = cv2.cvtColor(pp, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(imag2)
    # s += 50
    s = cv2.equalizeHist(s)
    v = cv2.equalizeHist(v)
    chs = [h, s, v]
    imgRes = cv2.merge(chs)
    # cv2.imshow("source",img)
    # cv2.imshow("imag2",imag2)
    # imgRes = imag2.copy()
    test = cv2.cvtColor(imgRes, cv2.COLOR_HSV2BGR)
    im = cv2.inRange(imgRes, (0, 100, 30), (15, 255, 255))
    im2 = cv2.inRange(imgRes, (160, 100, 30), (180, 255, 255))
    #### imgF = im + im2
    imgF = cv2.bitwise_or(im, im2)
    imP = cv2.inRange(prueba, (0, 100, 30), (15, 255, 255))
    imP2 = cv2.inRange(prueba, (160, 100, 30), (180, 255, 255))
    ##### imgFP = imP + imP2
    imgFP = cv2.bitwise_or(imP, imP2)
    # cv2.imshow("PRUEBA", test)
    #
    # printHSV_Values(imgRes)
    # printHSV_Values(prueba)

    ##### final = imgF + imgFP
    final = cv2.bitwise_or(imgF, imgFP)
    # cv2.imshow("PRUEBA2", test3)
    # cv2.imshow("pr",imP)
    # cv2.imshow("ORIGINAL", imP2)
    # cv2.imshow("imgF",imgF)
    # cv2.imshow("imFP",imgFP)
    # cv2.imshow("final",final)
    # cv2.waitKey(800)
    # cv2.destroyAllWindows()

    if (show):
        cv2.imshow("image", image)
        cv2.imshow("win1", im)
        cv2.imshow("win2", im2)
        cv2.imshow("win3", imgF)
        cv2.waitKey()
        cv2.destroyWindow("image")
        cv2.destroyWindow("win1")
        cv2.destroyWindow("win2")
        cv2.destroyWindow("win3")
    return final


def whiteAreaDetection(image, name, show=False):
    img = image.copy()
    test3 = getBinaryInvMask(img)
    pp = cv2.bitwise_and(img, img, mask=test3)
    # cambiar espacio rgb -> HSV
    img = image.copy()
    b, g, r = cv2.split(img)
    r = preProcessImage(r)
    g = preProcessImage(g)
    b = preProcessImage(b)
    rgb = [b, g, r]
    prc = cv2.merge(rgb)
    a = image.copy()
    hsv = cv2.cvtColor(a, cv2.COLOR_BGR2HSV)
    cv2.imshow("hsv", hsv)
    cv2.waitKey(800)
    cv2.destroyAllWindows()
    h, s, v = cv2.split(hsv)
    prueba = cv2.cvtColor(prc, cv2.COLOR_BGR2HSV)
    imag2 = cv2.cvtColor(pp, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(imag2)
    # s += 50
    s = cv2.equalizeHist(s)
    v = cv2.equalizeHist(v)
    chs = [h, s, v]
    imgRes = cv2.merge(chs)
    imTest = cv2.inRange(hsv, (0, 0, 70), (0, 10, 255))

    cv2.imshow("imgRes", imgRes)
    cv2.imshow("pp", pp)
    cv2.imshow("image", image)
    cv2.imshow("win3", imTest)
    cv2.waitKey(800)
    cv2.destroyAllWindows()

    if (show):
        cv2.imshow("image", image)
        cv2.imshow("win3", imTest)
        cv2.waitKey()
        cv2.destroyWindow("image")
        cv2.destroyWindow("win3")
    return imTest

def drawPoints(image, listPoints):
    for point in listPoints:
        x, y = point
        pt = (x, y)
        cv2.circle(image, pt, 5, (0, 255, 0), -1)
        cv2.imshow("Point in list", image)
        cv2.waitKey(400)
        cv2.destroyWindow("Point in list")
    cv2.imshow("Points in list", image)
    cv2.waitKey(1000)
    cv2.destroyWindow("Points in list")


def getAndDrawPoints(image, listIndex, list, ShowImage=True):
    for index in listIndex:
        pt = (list[index][0], list[index][1])
        cv2.circle(image, pt, 3, (255, 0, 0), -1)
    if (ShowImage):
        cv2.imshow("Points in list", image)
        cv2.waitKey()
        cv2.destroyAllWindows()


def find_lines(image, mask):
    #blur mask of image

    img = image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst = cv2.Canny(mask.copy(), 50, 200)
    #dst = cv2.Canny(mask.copy(),50,150,apertureSize=3)
    # cv2.imshow("canny", dst)
    # cv2.waitKey(500)
    # cv2.destroyWindow("canny")
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
            # cv2.imshow("line", img)
            # cv2.waitKey(500)
            # cv2.destroyWindow("line")
            punto = np.array([pt1, pt2])
            linesConverted.append(punto)
        # cv2.imshow("line", img)
        # cv2.waitKey(250)
        # cv2.destroyWindow("line")
        puntos = Vect.acumularPuntosInterseccion(np.asarray(linesConverted), image.copy())
        if (len(puntos) == 3):
            ## COMPROBAR QUE LAS LINEAS DE LOS PUNTOS DEL TRIANGULO SEAN AGUDOS EN TORNO A LOS 60º
            ## COMPROBAR SI HAY LINEAS PARALELAS ANTES
            ## HACER UN ACUMULADOR DE RESULTADOS ENTRE FONDO, TRIANGULOS, CIRCULOS PARA RESULTADO DE IMAGEN
            if (Vect.checkAngle(puntos)):
                return "triangle"
            # print("Is not a warning triangle shape")
            return None
        else:
            return None

    return None

def findCircles(img, param2, minRad=0, maxRad=0):
    try:
        # circles = cv2.HoughCircles(img.copy(), cv2.HOUGH_GRADIENT, 2, 1, np.array([]), 100, param2, 1)
        circles = cv2.HoughCircles(img.copy(), cv2.HOUGH_GRADIENT, 1, 1,
                                   param1=50, param2=param2, minRadius=minRad, maxRadius=maxRad)
    except:
        circles = None

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
    cv2.destroyWindow("imagen")
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
    listCircles = []
    find = False
    while (param2 > minParam2):
        circles = findCircles(blurMask, param2)
        if (isinstance(circles, str)):
            param2 = param2 - 1
        elif (not (circles is None)):

            a, b, c = circles.shape
            for i in range(b):
                ptX, ptY, r = circles[0][i]
                # print ((r,rm))
                if (not (listCircles.__contains__((ptX, ptY, r)))):

                    # print("hay circulos, paramatro2 valor = ", param2)


                    cv2.circle(ccimg, (ptX, ptY), 2, (0, 255, 0), 3)
                    cv2.circle(ccimg, (ptX, ptY), r, (255, 0, 0), 5)

                    c1 = [int(ptX - r - 3), int(ptY - r - 3)]
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

                    cv2.rectangle(ccimg, (c1[0], c1[1]), (c4[0], c4[1]), (255, 0, 0), 3)

                    # COMPROBAR SI EL FRAGMENTO DE LA MASCARA TIENE MAS DE 500 puntos
                    # cimg = cv2.resize(cimg,(x*2,y*2))
                    # path = ruta.split("\\")
                    # print(path[-1])
                    ############################
                    if (overflow):
                        ccimg = cimg.copy()
                        overflow = False
                        continue
                    ############################
                    # cv2.imshow("original", img)
                    # cv2.imshow("mask", blurMask)
                    # cv2.imshow("detected circles", ccimg)
                    # cv2.waitKey(500)
                    # cv2.destroyWindow("original")
                    # cv2.destroyWindow("mask")
                    # cv2.destroyWindow("detected circles")
                    # print("----> Circulo detectado en la imagen : ", path, " <------")
                    ##en un futuro hay que obtener todos los circulos detectados, ordernar y obtener el circulo
                    ## con mayor area, osea el mas grande :D
                    listCircles.append((ptX, ptY, r))
                    find = True
                    if (r > 60):
                        return "circle"

        # if (param2 <= minParam2):
        #     # no se han detectado circulos, pasamos a detectar lineas
        #     # print("couldn't find important shape")
        #     show = cv2.resize(img.copy(), (s[0] * 2, s[1] * 2))
        #     path = ruta.split("\\")
        #     # print(path[-1])
        #     return None
        param2 -= 1
    if (find):
        ccimg = cimg.copy()
        circ = listCircles.copy()
        circ = sorted(circ, key=lambda c: c[2])
        bigger = circ[-1]
        ptX, ptY, r = bigger
        # cv2.circle(ccimg, (ptX, ptY), 2, (0, 255, 0), 3)
        # cv2.circle(ccimg, (ptX, ptY), r, (255, 0, 0), 5)
        c1 = [int(ptX - r - 3), int(ptY - r - 3)]
        c4 = [int(ptX + r + 3), int(ptY + r + 3)]
        # cv2.rectangle(ccimg, (c1[0], c1[1]), (c4[0], c4[1]), (255, 0, 0), 3)
        if (r >= 60):
            # cv2.imshow("detected circles", ccimg)
            # cv2.waitKey(200)
            # cv2.destroyWindow("detected circles")
            #

            return "circle"
    return None

def shapeDetection(img, ruta):
    nombre = ((ruta.split("\\"))[-1])
    imageShape = img.shape
    Icopy = img.copy()
    Icp = Icopy.copy()
    # Icblur = cv2.medianBlur(Icopy, 7)
    Icblur = cv2.blur(Icp, (5, 5))
    redMask = redAreaDetection(Icblur, nombre)
    mask = redMask.copy()
    shape = mask.shape
    s = [shape[0], shape[1]]
    rm = (np.amin(s) / 3)
    rmax = (np.amax(s) / 2)
    res = np.ones_like(Icopy.copy())
    thr = int(shape[0] / 8)
    blurMask = cv2.medianBlur(mask, 3)
    blurMask = cv2.blur(blurMask.copy(), (9, 9))
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(blurMask.copy(), cv2.MORPH_OPEN, kernel, iterations=3)
    kernelEro = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(blurMask.copy(), kernel, iterations=2)
    opening = cv2.GaussianBlur(opening, (9, 9), 0)
    erosion = cv2.GaussianBlur(erosion, (9, 9), 0)
    # cv2.imshow("test",opening)
    # cv2.imshow("blur", blurMask)
    # # cv2.moveWindow("test2",0,0)
    # cv2.waitKey(800)
    # cv2.destroyAllWindows()
    ##
    # blurMask = cv2.resize(blurMask, (imageShape[0] * 5, imageShape[1] * 5))
    # res = cv2.bitwise_and(Icp, Icp, mask=blurMask)

    # cv2.imshow("original", img)
    # cv2.imshow("source", Icopy)
    # cv2.imshow("bitwise", res)
    # cv2.imshow("blur", blurMask)
    # cv2.imshow("opening", opening)
    # cv2.imshow("erosion", erosion)
    # cv2.waitKey(1000)
    # cv2.destroyAllWindows()
    cimg = Icopy.copy()  # numpy function
    # circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 10, np.array([]), 100, 30, 1, 30)
    NCircles = circles(img.copy(), opening.copy(), cimg, ruta)
    if NCircles is None:
        return find_lines(Icopy.copy(), erosion.copy())
    else:
        return NCircles
