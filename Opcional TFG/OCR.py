from operator import itemgetter
from os import listdir
from os import path
from os import remove

import cv2
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# from sklearn.lda import LDA
clasificador_knear = None
Lda = None

def preProcessImage(image):
    channel = image.copy()
    # channel = cv2.adaptiveThreshold(channel, 255, adaptive_method=cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=55, param1=7)
    # channel = cv2.adaptiveThreshold(channel,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,55,7)
    channel = cv2.adaptiveThreshold(channel, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 55, 3)
    # mop up the dirt
    channel = cv2.dilate(channel, None, 1)
    channel = cv2.erode(channel, None, 1)
    return channel


def preProcessImage2(image, stop):
    channel = image.copy()
    # channel = cv2.adaptiveThreshold(channel, 255, adaptive_method=cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=55, param1=7)
    # channel = cv2.adaptiveThreshold(channel,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,55,7)
    channel = cv2.adaptiveThreshold(channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 61, 3)
    # mop up the dirt
    channel = cv2.dilate(channel, None, 1)
    channel = cv2.erode(channel, None, 1)
    return channel

def getBinaryInvMask(RGBImage, stop=False):
    img = RGBImage.copy()
    if (stop):
        gr = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
        gr = cv2.medianBlur(gr, 15)
        b, g, r = cv2.split(img)
        r = preProcessImage2(r, stop)
        g = preProcessImage2(g, stop)
        b = preProcessImage2(b, stop)
        rgb = [b, g, r]
        processedImage = cv2.merge(rgb)
        prp = processedImage.copy()
        grayImage = cv2.cvtColor(prp, cv2.COLOR_BGR2GRAY)
        ret, threshold = cv2.threshold(grayImage.copy(), 127, 255, cv2.THRESH_BINARY_INV)
        th2 = threshold.copy()
        threshold = cv2.dilate(threshold, (8, 8), iterations=5)
        threshold2 = cv2.medianBlur(threshold.copy(), 3)
        th3 = cv2.medianBlur(threshold.copy(), 5)
        # cv2.imshow("threshold2", threshold2)
        # cv2.imshow("adaptative", th3)
        # cv2.imshow("2", th2)
        # cv2.waitKey(800)
        # cv2.destroyAllWindows()
        return th3, th2
    else:

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



def preProcessImage(image):
        channel = image.copy()
        # channel = cv2.adaptiveThreshold(channel, 255, adaptive_method=cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=55, param1=7)
        # channel = cv2.adaptiveThreshold(channel,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,55,7)
        channel = cv2.adaptiveThreshold(channel, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 55, 3)
        # mop up the dirt
        channel = cv2.dilate(channel, None, 1)
        channel = cv2.erode(channel, None, 1)
        return channel

def preProcessImage2(image, stop):
        channel = image.copy()
        # channel = cv2.adaptiveThreshold(channel, 255, adaptive_method=cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=55, param1=7)
        # channel = cv2.adaptiveThreshold(channel,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,55,7)
        channel = cv2.adaptiveThreshold(channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 61, 3)
        # mop up the dirt
        channel = cv2.dilate(channel, None, 1)
        channel = cv2.erode(channel, None, 1)
        return channel


pathTrain = "./Training_Images/OCRImages/Train/training_ocr/"
pathToSave = "./training_file.npz"
# pathTrain = "./Training_Images/OCRImages/Train/imagenes/"
caracterMatrix = []
labelMatrix = []
carIndex = []
clf = GaussianNB()
numThreads = 4
diccionario = np.chararray((37, 1))
diccionario = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
               '7': 7, '8': 8, '9': 9, 'A': 10, 'B': 11, 'C': 12, 'D': 13,
               'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19, 'K': 20,
               'L': 21, 'M': 22, 'N': 23, 'O': 24, 'P': 25, 'Q': 26, 'R': 27,
               'S': 28, 'T': 29, 'U': 30, 'V': 31, 'W': 32, 'X': 33, 'Y': 34,
               'Z': 35}

trainShape = []
pruebaC = None
# '0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','ESP'

threads = []


def trainAndTest(image, limitShape, numericOCR=False, overwrite=True, train_path=None, savePath=None):
    global clasificador_knear
    global Lda
    if (train_path is None):
        train_path = pathTrain
    if (savePath is None):
        savePath = pathToSave
    caracterVector, labelVector = OCRTrain(train_path=train_path, avoidNumbers=(not numericOCR),
                                           overwriteFile=overwrite, savePathFile=savePath)
    vector_caracteristicas,Lda = usarLDA(caracterVector,labelVector)
    clasificador_knear = crearKNEAR(vector_caracteristicas,labelVector)
    if (numericOCR):
        result = searchDigits(image, limitShape, vector_caracteristicas, labelVector)
        print(result)
    else:
        result = searchStop(image, limitShape, caracterVector, labelVector)
        print(result)
    return result


def OCRTrain(train_path, avoidNumbers, overwriteFile=False, savePathFile=pathToSave):
    archivos = listdir(train_path)

    if not avoidNumbers:
        caracterMatrix, labelMatrix = train(archivos,True)
    else:
        caracterMatrix, labelMatrix = train(archivos)

    return caracterMatrix, labelMatrix


def isNumber(prime):

        if(not (
                prime == 'A' or
                prime == 'B' or
                prime == 'C' or
                prime == 'D' or
                prime == 'E' or
                prime == 'F' or
                prime == 'G' or
                prime == 'H' or
                prime == 'I' or
                prime == 'J' or
                prime == 'K' or
                prime == 'L' or
                prime == 'M' or
                prime == 'N' or
                prime == 'O' or
                prime == 'P' or
                prime == 'Q' or
                prime == 'R' or
                prime == 'S' or
                prime == 'T' or
                prime == 'U' or
                prime == 'V' or
                prime == 'W' or
                prime == 'X' or
                prime == 'Y' or
                prime == 'Z'
                )
        ):
            return True

        return False


def train(listFiles, onlyNumbers=False):
    for file in listFiles:
        prime = file.split('_')[0]
        if (onlyNumbers):
            if (not (isNumber(prime))):
                continue
        else:
            if (isNumber(prime)):
                continue

        if (prime == "ESP"):
            continue
        full_path = pathTrain + file
        img = cv2.imread(full_path, 0)
        image = cv2.imread(full_path)

        ret, umbral = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
        im2, contours, hierarchy = cv2.findContours(umbral.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        for cnt in contours:
            carIndex.append(diccionario[file.split('_')[0]])
            x, y, w, h = cv2.boundingRect(cnt)
            im = image.copy()
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 1)
            umbralcut = umbral[y:y + h, x:x + w]
            caracter = cv2.resize(umbralcut, (10, 10), None, 0, 0, cv2.INTER_NEAREST)
            caracterLine = np.asarray(caracter).ravel()
            # letralinea.reshape(-1,1)
            # caracterLine = caracterLine.reshape(-1,1)
            caracterMatrix.append(caracterLine)
            labelMatrix.append(ord(prime))

            # cv2.imshow("caracter",im)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
            # cv2.imshow("umbral",umbral)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
            # cv2.imshow("umbralCut",umbralcut)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
    cm = caracterMatrix
    lbl = labelMatrix

    return cm, lbl



def searchStop(image, limitShape, caracterList, labelList):
    # imgR=cv2.resize(image.copy(),(200,200),interpolation=cv2.INTER_LANCZOS4)
    buffer = ['S', 'T', 'O', 'P']
    ranX = 5
    ranY = 5
    print("Searching STOP")
    classifier = GaussianNB()
    classifierKNN = KNeighborsClassifier(n_neighbors=1)
    classifierKNN.fit(caracterList, labelList)
    classifier.fit(caracterList, labelList)
    img2 = image.copy()
    img = image.copy()
    prueba = img.copy()
    # prueba = cv2.medianBlur(prueba, 9)
    Sx, Sy, ch = image.shape
    print("shape = ", Sx, Sy)
    print("mid ", Sx // 2, Sy // 2)
    midshape = (Sx // 2, Sy // 2)
    secTercShape = (2 * Sx // 3 + 1, 2 * Sy // 3 + 1)
    primTercShape = (Sx // 3, Sy // 3)
    mask, th = getBinaryInvMask(prueba, True)
    # cv2.imshow("threshold", th)
    # cv2.imshow("mask", mask)
    # cv2.waitKey(800)
    # cv2.destroyAllWindows()
    # cv2.line(img,(0,primTercShape[1]),(x,primTercShape[1]),(0,255,0),thickness=1)
    # cv2.line(img,(0,secTercShape[1]),(x,secTercShape[1]),(0,255,0),thickness=1)
    imgContours, npaContours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    lenContours = len(npaContours)
    validContours = []
    imaa = img.copy()
    if (len(limitShape) == 3):
        L2, L3, L4 = limitShape
        listY = [L2[0], L3[0], L4[0]]
        listX = [L2[1], L3[1], L4[1]]
        L2 = np.amin(listY)
        L4 = np.amax(listY)
        L3 = np.amax(listX)
    else:
        L1, L2, L3, L4 = limitShape
    cv2.line(img2, (0, L2), (L3, L2), (255, 0, 0), 2)
    cv2.line(img2, (0, L4), (L3, L4), (0, 125, 255), 2)
    # cv2.imshow("rect", img2)
    # cv2.waitKey(800)
    # cv2.destroyAllWindows()
    for i in range(0, lenContours):
        cnt = npaContours[i]
        ord = hierarchy[0][i][2]
        x, y, w, h = cv2.boundingRect(cnt)
        ima = img[y:y + h, x:x + w]
        # cv2.imshow("ima", ima)
        # cv2.waitKey(800)
        # cv2.destroyWindow("ima")
        ptoX, ptoY = x + w, y + h
        if ((w * h) > 1000 and (w * h) <= 3000):
            if (ptoY >= L2 and ptoY < L4):  # y > primTercShape[1]):

                if (1):
                    if (1):  # ord != -1):
                        # print(hierarchy[0][i])
                        # print(" --- punto X,Y ", (x, y), " --- Punto X+W , Y+H --- ", (x + w, y + h))
                        # cv2.imshow("ima", ima)
                        # cv2.waitKey(800)
                        # cv2.destroyWindow("ima")
                        # if(y>=60 and (w*h)>2000):
                        # print(w*h)
                        # print(x,y,w,h)
                        validContours.append([x, y, w, h])
                        # else:continue

    validContours2 = sorted(validContours, key=itemgetter(0))
    cnt = 0
    kernel = np.ones((3, 3), np.uint8)
    for x, y, w, h in validContours2:
        if (len(buffer) <= 0):
            break
        cv2.rectangle(img2, (x - 20, y - 20), (x + w + 20, y + h + 20), (0, 255, 126), 3)
        puntoY = y - ranY
        puntoX = x - ranX
        if (puntoX < 0): puntoX = 0
        if (puntoY < 0): puntoY = 0
        umbral = th[puntoY:y + h + ranY, puntoX:x + w + ranX]
        # umbral = cv2.GaussianBlur(umbral,(7,7),0)
        # umbral = cv2.medianBlur(umbral, 3)
        umbral = cv2.medianBlur(umbral, 9)
        umbral = cv2.dilate(umbral, (3, 3), iterations=3)
        testImage = umbral
        caracterTest = cv2.resize(testImage, (10, 10), None, 0, 0, cv2.INTER_NEAREST)
        caracterLine = np.asarray(caracterTest.ravel())
        caracterLine = caracterLine.reshape(1, -1)
        # res = classifier.predict(caracterLine)
        res = classifierKNN.predict(caracterLine)
        res2 = classifier.predict(caracterLine)
        # cv2.imshow("imagen", img2)
        # cv2.imshow("umbral", umbral)
        # cv2.waitKey(800)
        # cv2.destroyAllWindows()
        print("el resultado es", res)

        try:
            index = buffer.index(res)
        except:
            index = -1
        if (index != -1):
            # print("el resultado es", res)
            print(x, y, w, h)
            cnt += 1
            buffer.remove(res)
        # cv2.imshow("char",testImage)
        # cv2.waitKey(1000)
        # cv2.destroyAllWindows()

    leng = len(buffer)
    if (leng > 0):  # not all characters detected
        if (cnt >= 3):  # almost all detected
            return "stop"
    else:
        return "stop"


def findCircles(img, param2, minRad=0, maxRad=0):
    try:
        # circles = cv2.HoughCircles(img.copy(), cv2.HOUGH_GRADIENT, 2, 1, np.array([]), 100, param2, 1)
        circles = cv2.HoughCircles(img.copy(), cv2.HOUGH_GRADIENT, 1, 1,
                                   param1=50, param2=param2, minRadius=minRad, maxRadius=maxRad)
    except:
        circles = None

    return circles


def searchDigits(image, limitShape, caracterList, labelList):
    global clasificador_knear
    global Lda
    recognition = []


    img2 = image.copy()
    img = image.copy()
    prueba = img.copy()
    prueba = cv2.medianBlur(prueba, 9)
    Sx, Sy, ch = image.shape
    print("shape = ", Sx, Sy)
    print("mid ", Sx // 2, Sy // 2)
    midshape = (Sx // 2, Sy // 2)
    secTercShape = (2 * Sx // 3 + 1, 2 * Sy // 3 + 1)
    primTercShape = (Sx // 3, Sy // 3)
    mask = getBinaryInvMask(prueba)
    redMask = redAreaDetection(prueba,"",False)
    circles = None
    param2 = 100
    while circles is None:
        circles = findCircles(redMask, param2)
        param2 -= 1
    # mask=cv2.bitwise_not(mask)

    circles = np.round(circles[0, :]).astype("int")
    shape = img.shape
    # loop over the (x, y) coordinates and radius of the circles
    for (x, y, r) in circles:

        # cv2.circle(img, (x, y), r, (0, 255, 0), 4)
        # cv2.rectangle(img, (x, y), (x, y), (0, 128, 255), -1)
        c1 = [int(x - r), int(y - r)]
        c4 = [int(x + r), int(y + r)]
        if c1[0] < 0: c1[0] = 0
        if c1[1] < 0: c1[1] = 0
        if c4[0] > shape[0]: c4[0] = shape[0]-5
        if c4[1] > shape[1]: c4[1] = shape[1]-5

        # cv2.rectangle(img, (c1[0], c1[1]), (c4[0], c4[1]), (255, 0, 0), 3)
    imagenDeteccion = img.copy()
    if circles is not None:
        imagenDeteccion = img[c1[1]:c4[1], c1[0]:c4[0]]
    # cv2.imshow("output", np.hstack([image, img]))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imshow("mask", mask)
    # cv2.waitKey(800)
    # cv2.destroyAllWindows()
    cv2.imshow("circ",imagenDeteccion)
    cv2.waitKey(800)
    cv2.destroyAllWindows()
    gray = cv2.cvtColor(imagenDeteccion, cv2.COLOR_BGR2GRAY)
    imgBlurred = cv2.GaussianBlur(gray, (5,5), 0)
    # imgBlurred = cv2.GaussianBlur(mask, (5, 5), 0)
    ret, th = cv2.threshold(mask, 120, 255, cv2.THRESH_BINARY_INV)
    # th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # ret, th = cv2.threshold(redMask, 127, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("mask", th)
    # cv2.imshow("imagen",img)
    cv2.waitKey(800)
    cv2.destroyAllWindows()
    imgContours, npaContours, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    lenContours = len(npaContours)
    validContours = []
    for i in range(0, lenContours):
        cnt = npaContours[i]
        ord = hierarchy[0][i][2]
        x, y, w, h = cv2.boundingRect(cnt)
        xw = x+w
        yh = y+h
        ima = img[y:y + h, x:x + w]
        # if ((w * h) > 700 and (w * h) <= 2500):
        #     if (y > primTercShape[1]):
        ptoX, ptoY = x + w, y + h
                # if (1):
                #     if (1):  # ord != -1):
        print(hierarchy[0][i])
        print(" --- punto X,Y ", (x, y), " --- Punto X+W , Y+H --- ", (x + w, y + h))
        # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0))
        cv2.imshow("ima", ima)
        cv2.waitKey(800)
        cv2.destroyWindow("ima")
        # if(y>=60 and (w*h)>2000):
        # print(w*h)
        # print(x,y,w,h)
        validContours.append([x, y, w, h])
         # else:continue

# HAY QUE RETOCAR LAS JERARQUIAS PARA OBTENER LOS DIGITOS QUE ESTAN DENTRO DE UNA ZONA Y SON UNICOS, DA IGUAL SI TIENEN HIJOS AKA MISMO TAMAÑO
    cv2.imshow("ima", img)
    cv2.waitKey(800)
    cv2.destroyWindow("ima")
    validContours2 = sorted(validContours, key=itemgetter(0))
    cnt = 0
    for x, y, w, h in validContours2:

        cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 126), 3)
        # testImage = gray[y:y + h, x:x + w]
        umbral = img[y:y + h, x:x + w]
        # umbral = cv2.medianBlur(umbral, 9)
        # umbral = cv2.dilate(umbral, (3, 3), iterations=3)
        # cv2.imshow("imagen", img2)
        # cv2.imshow("char", img)
        cv2.imshow("umbral", umbral)
        cv2.waitKey(800)
        cv2.destroyAllWindows()
        testImage = cv2.cvtColor(umbral,cv2.COLOR_BGR2GRAY)
        ret, testImage = cv2.threshold(testImage, 150, 255, cv2.THRESH_BINARY)
        cv2.imshow("threshold",testImage)
        cv2.waitKey(800)
        cv2.destroyAllWindows()
        caracterTest = cv2.resize(testImage, (10, 10), None, 0, 0, cv2.INTER_NEAREST)
        caracterLine = np.asarray(caracterTest).ravel()
        # caracterLine = caracterLine.reshape(1, -1)
        letra = Lda.transform(caracterLine)
        ret, results, neighbours, dist = clasificador_knear.findNearest(np.float32(letra),10)
        res = chr(int(ret))
        recognition.append(res)
        # res = classifierKNN.predict(caracterLine)
        print("el resultado es", res)

def usarLDA(array,e):
    Lda = lda()
    caracteres = np.vstack(array)
    E = np.array(e)
    Lda.fit_transform(caracteres,E)
    CR = Lda.transform(caracteres)
    CR = CR.astype(np.float32, copy=True)
    return CR,Lda

def crearKNEAR(CR,e):
    E = np.array(e)
    Knear = cv2.ml.KNearest_create()
    Knear.train(CR,cv2.ml.ROW_SAMPLE,E)
    return Knear


#
# caracterMatrix, labelMatrix = entrenarOCR()
# image = cv2.imread("./auxiliar_images/stopsign2.jpg")

# image = cv2.imread("./Training_Images/Training/Images_Sign_Detection_Benchmark\\14\\00001.ppm")
# image2 = cv2.resize(image.copy(), (200, 200))
# lista = [45, 46, 195, 196]
# res = trainAndTest(image2, lista, overwrite=False)
# print (res)

image = cv2.imread("./Training_Images/Training/Images_Sign_Detection_Benchmark\\01\\00001.ppm")
image2 = cv2.resize(image.copy(), (200, 200))
image_shape = image2.shape
lista = [0,0,image_shape[0],image_shape[1]]
res = trainAndTest(image2,lista,True)
print(res)
# resultado = searchStop(image2, caracterMatrix, labelMatrix)
# print(resultado)
