import cv2
import numpy as np

from auxiliar_clases import mathFunctions as Vect


def redAreaDetection(image, show=False):
    # image2 = cv2.imread("./auxiliar_images/cirRoj.jpg")
    img = image.copy()

    # cambiar espacio rgb -> HSV
    imag2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    x, y, ch = img.shape
    cv2.imshow("hsvor", imag2)
    h, s, v = cv2.split(imag2)
    # s=10*s
    s = cv2.equalizeHist(s)
    v = 10 * v
    chs = [h, s, v]
    imgRes = cv2.merge(chs)
    hsv_1 = (110, 50, 50)
    hsv_2 = (130, 255, 255)
    # im = cv2.inRange(imag2,(0,50,50),(20,255,255))
    im = cv2.inRange(imgRes, (0, 40, 40), (10, 255, 255))
    im2 = cv2.inRange(imgRes, (160, 40, 40), (179, 255, 255))
    imgF = im  # +im2

    cv2.imshow("win3", imgF)
    cv2.destroyAllWindows()
    cv2.imshow("hsv", im2)
    cv2.destroyAllWindows()

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
        cv2.circle(image, pt, 3, (255, 0, 0), -1)
    cv2.imshow("Points in list", image)
    cv2.waitKey()
    cv2.destroyAllWindows()


def getAndDrawPoints(image, listIndex, list, ShowImage=True):
    for index in listIndex:
        pt = (list[index][0], list[index][1])
        cv2.circle(image, pt, 3, (255, 0, 0), -1)
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


def find_lines(image):
    #blur mask of image

    img = image.copy()
    dst = cv2.Canny(img.copy(), 50, 200, apertureSize=3)
    cv2.imshow("canny", dst)
    cv2.waitKey(500)
    cv2.destroyAllWindows()
    px, py = img.shape
    m = [px, py]
    threshold = int(np.amin(m) / 2)
    while (threshold > 0):
        lines = cv2.HoughLines(dst, 1, np.pi / 180, threshold)
        if (lines is None):
            threshold -= 1
        elif (len(lines) > 2):
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
                # cv2.line(src, pt1, pt2, (255, 0, 0), 1, cv2.LINE_AA)
                # cv2.imshow("line", src)
                # cv2.waitKey(200)
                # cv2.destroyAllWindows()
                punto = np.array([pt1, pt2])
                linesConverted.append(punto)
                # cv2.imshow("mm",src)
                # # cv2.waitKey(2000)
                # cv2.destroyAllWindows()
                puntos = Vect.acumularPuntosInterseccion(np.asarray(linesConverted), img)


def findCircles(img, param2, minRad, maxRad):
    try:
        # circles = cv2.HoughCircles(img.copy(), cv2.HOUGH_GRADIENT, 2, 1, np.array([]), 100, param2, 1)
        circles = cv2.HoughCircles(img.copy(), cv2.HOUGH_GRADIENT, 1, 1,
                                   param1=100, param2=param2, minRadius=0, maxRadius=0)
    except:
        circles = "nada"

    return circles


def lineas(imageName):
    fn = imageName
    # src = cv2.imread(fn)
    src = fn.copy()
    imShape = fn.shape
    bn = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    dst = cv2.Canny(bn, 50, 200, apertureSize=3)
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
        #pruebaCirculo(src, None)
        return

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
        # cv2.line(src, pt1, pt2, (255, 0, 0), 1, cv2.LINE_AA)
        # cv2.imshow("line", src)
        # cv2.waitKey(200)
        # cv2.destroyAllWindows()
        punto = np.array([pt1, pt2])
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


def shapeDetection(img):
    imageShape = img.shape
    Icopy = cv2.resize(img.copy(), (imageShape[0] * 2, imageShape[1] * 2))

    redMask = redAreaDetection(Icopy)
    if (not (redMask is None)):
        print("Hay zona roja")
        print("Detectar circulos")
        mask = redMask.copy()
        shape = mask.shape
        s = [shape[0], shape[1]]
        rm = (np.amin(s) / 3)
        rmax = (np.amax(s) / 2)
        res = np.ones_like(Icopy.copy())
        thr = int(shape[0] / 8)
        blurMask = cv2.medianBlur(mask, 5)

        # blurMask = cv2.GaussianBlur(mask,(5,5),0)

        # dst = cv2.Canny(blurMask.copy(), 400, 500, apertureSize=3)
        # # dst = cv2.Canny(dst, 50, 200, apertureSize=3)
        # cv2.imshow("canny",dst)

        # cv2.waitKey(500)
        # cv2.destroyAllWindows()
        # cv2.imshow("blur",blurMask)
        # cv2.imshow("original",redMask)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        cimg = Icopy.copy()  # numpy function
        # circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 10, np.array([]), 100, 30, 1, 30)
        param2 = 100
        while (param2 > 0):
            circles = findCircles(blurMask, param2, rm, rmax)
            if (isinstance(circles, str)):
                param2 = param2 - 1
            elif (not (circles is None)):
                # print("hay circulos, paramatro2 valor = ",param2)
                a, b, c = circles.shape
                for i in range(b):
                    ptX, ptY, r = circles[0][i]
                    # print ((r,rm))
                    if (r >= rm and r <= rmax):
                        cv2.circle(cimg, (ptX, ptY), 2, (0, 255, 0), 3)
                        cv2.circle(cimg, (ptX, ptY), r, (0, 0, 255), 2)
                        x, y, ch = cimg.shape
                        # cimg = cv2.resize(cimg,(x*2,y*2))
                        cv2.imshow("original", img)
                        cv2.imshow("mask", blurMask)
                        cv2.imshow("detected circles", cimg)
                        cv2.waitKey(500)
                        cv2.destroyAllWindows()
                        return "circle"

            param2 -= 1
        if (param2 == 0):
            # no se han detectado circulos, pasamos a detectar lineas
            print("no se han detectado circulos, pasamos a detectar lineas en busca de triangulos")
            cv2.imshow("imagen", img)
            cv2.imshow("blurMask", blurMask)
            cv2.waitKey()
            cv2.destroyAllWindows()
            points = find_lines(redMask)
            return ("triangle")
    else:
        print("No se detecta zona roja")
        return ("other")
