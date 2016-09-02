from operator import itemgetter
from os import listdir
from os import path
from os import remove

import cv2
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from auxiliar_clases import graphic_features as graph

# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# from sklearn.lda import LDA


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


def trainAndTest(image, numericOCR=False, overwrite=True, train_path=None, savePath=None):
    if (train_path is None):
        train_path = pathTrain
    if (savePath is None):
        savePath = pathToSave
    caracterVector, labelVector = OCRTrain(train_path=train_path, avoidNumbers=(not numericOCR),
                                           overwriteFile=overwrite, savePathFile=savePath)
    if (numericOCR):
        result = searchDigits(image, caracterVector, labelVector)
        print(result)
    else:
        result = searchStop(image, caracterVector, labelVector)
        print(result)
    return result


def OCRTrain(train_path, avoidNumbers, overwriteFile=False, savePathFile=pathToSave):
    print("OCR function training")
    if (not savePathFile.endswith('.npz')):
        savePathFile += ".npz"
    if (not overwriteFile):
        caracterMatrix, labelMatrix = load(savePathFile)
    if ((overwriteFile) or (caracterMatrix is None) or (labelMatrix is None)):
        print("training_file not found or overwrite flag set True, let's train :>")
        if (path.exists(savePathFile)):
            remove(savePathFile)
        archivos = listdir(train_path)
        # caracterMatrix, labelMatrix = tratamiento(0, len(archivos), archivos)
        caracterMatrix, labelMatrix = train(archivos)  # tratamientoV2(archivos)

        save(caracterMatrix, labelMatrix, savePathFile)
    print("Training ended")
    return caracterMatrix, labelMatrix

def load(pathFile="./training_file.npz"):
    if (not pathFile.endswith('.npz')):
        pathFile += ".npz"
    try:
        data = np.load(pathFile)
        caracterMatrix = data['caracter']
        labelMatrix = data['label']
        return caracterMatrix, labelMatrix
    except:
        return None, None


def save(caracterMatrix, labelMatrix, pathFile="./training_file.npz"):
    np.savez(pathFile, caracter=caracterMatrix, label=labelMatrix)


def isNumber(elem):
    try:
        number = int(elem)
        return True
    except:
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
            caracterLine = np.asarray(caracter)  # .ravel()
            caracterLine = np.asarray(caracterLine.ravel())
            # letralinea.reshape(-1,1)
            # caracterLine = caracterLine.reshape(-1,1)
            caracterMatrix.append(caracterLine)
            labelMatrix.append(prime)

            # cv2.imshow("caracter",im)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
            # cv2.imshow("umbral",umbral)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
            # cv2.imshow("umbralCut",umbralcut)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
    cm = np.asarray(caracterMatrix)
    lbl = np.asarray(labelMatrix)
    return cm, lbl


def analizarImagen(imag):
    imagen = cv2.resize(imag, (500, 500))
    a, b, ch = imagen.shape
    mid = [int(a / 2), int(b / 2)]
    print(a, b)
    shp = [a, b]
    minimum = int((np.amin(shp)) / 10)
    print(minimum)
    maximum = int((np.amin(shp)) / 2)
    print(maximum)
    print("busca texto STOP en la imagen")
    carMatrix = np.vstack(caracterMatrix)
    matr = []
    x2 = 0
    x3 = 0
    y3 = 0
    y2 = 0
    E = np.array(carIndex)
    # clf.fit_transform(carMatrix, carIndex)
    # CR = clf.transform(carMatrix)
    # CR = CR.astype(np.float32, copy=True)
    CR = carMatrix.astype(np.float32, copy=True)
    E = E.reshape((E.size, 1))
    # Bayes = cv2.NormalBayesClassifier(CR,responses=E)
    # Knear = cv2.KNearest(CR, E, max_k=50)
    Knear = cv2.ml.KNearest_create()
    Knear.train(CR, cv2.ml.ROW_SAMPLE,E)
    # em = cv2.EM()
    # em.train(CR,labels=E)
    img = imagen.copy()
    imgBN = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBN = cv2.medianBlur(imgBN, 5)
    cv2.imshow("imagen", imgBN)
    cv2.waitKey(800)
    cv2.destroyAllWindows()
    imgTH = cv2.adaptiveThreshold(imgBN, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    imgTH = cv2.medianBlur(imgTH, 7)
    cv2.imshow("imagen", imgTH)
    # cv2.imshow("imagen2",imgTH2)
    cv2.waitKey(800)
    cv2.destroyAllWindows()
    im2, contours, hierarchy = cv2.findContours(imgTH.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    cntOrd = []
    parents = []
    sections = []
    subContours = []
    subHierarchy = []
    rectangulos = []
    print(len(hierarchy[0]))
    for i in range(0, contours.__len__()):
        x, y, w, h = cv2.boundingRect(contours[i])

        distX = np.abs(w - x)
        distY = np.abs(h - y)
        area = distX * distY

        if (w > minimum and h > minimum):
            if (w < maximum and h < maximum):
                ms = mid[1] + maximum, mid[1] - maximum
                if (h <= ms[0] and h >= ms[1]):
                    # cv2.rectangle(img,(x,y),(x+w,y+h),(50,255,50),1)
                    sections.append([x, y, w, h])
                    subContours.append(contours[i])
                    subHierarchy.append(hierarchy[0][i])
                    # print("valor de w y h: ", (w,h))

                    hi = hierarchy[0][i][2]
                    if (hi != -1):
                        x, y, w, h = cv2.boundingRect(contours[i])
                        cv2.rectangle(imagen, (x, y), (x + w, y + h), (50, 255, 50), 1)

                        rectangulos.append([x, y, w, h])
                        # if (h > (y3 - y2) / 2 and w > (x3 - x2) / 2):
                        #     parents.append(i)
    # sorted(student_tuples, key=lambda student: student[2])





    if (True):
        ind = 0
        results = None
        for rect in rectangulos:
            # letra = imgTH[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
            letra = imgTH[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
            cv2.imshow("letra", letra)
            cv2.waitKey(200)
            cv2.destroyAllWindows()
            caracter = cv2.resize(letra, (10, 10), None, 0, 0, cv2.INTER_NEAREST)
            # print(caracter.shape)
            letralinea = np.asarray(caracter).ravel()
            # letralinea=letralinea.reshape(-1,1)
            letralinea = clf.transform(letralinea)
            letralinea = letralinea.astype(np.float32)
            # retval, results = Bayes.predict(np.float32(letralinea))
            retval, results, neighbors, dists = Knear.findNearest(letralinea, 1)
            # retval,probs = em.predict(np.float32(letralinea))
            print(retval)
            for key, value in diccionario.items():
                if value == int(retval):
                    # print key
                    if value != 36:
                        matr.append(key)
                        print(key)
                        cv2.putText(imagen, key, (rect[0] + x2, rect[1] - 20 + y2), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255))
                    # cv2.destroyAllWindows()
                    break
                    # print retval
        if (not results is None):
            print(results)
            if (matr.__contains__("STP")):
                print(True)

            cv2.circle(img, (mid[0], mid[1]), 3, (0, 0, 0), -1)
            cv2.imshow("mmm", img)
            cv2.waitKey()
            cv2.destroyAllWindows()
            cv2.imshow("imagen", imagen)
            cv2.waitKey()
            cv2.destroyAllWindows()


def searchStop(image, caracterList, labelList):
    # imgR=cv2.resize(image.copy(),(200,200),interpolation=cv2.INTER_LANCZOS4)
    buffer = ['S', 'T', 'O', 'P']
    print("Searching STOP")
    classifier = GaussianNB()
    classifierKNN = KNeighborsClassifier(n_neighbors=1)
    classifierKNN.fit(caracterList, labelList)
    # classifier.fit(caracterList,labelList)
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
    mask = graph.getBinaryInvMask(prueba)
    # mask=cv2.bitwise_not(mask)
    cv2.imshow("source", image)
    cv2.imshow("mask", mask)
    cv2.waitKey(800)
    cv2.destroyAllWindows()
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    # imgBlurred = cv2.GaussianBlur(gray, (5,5), 0)
    # imgBlurred = cv2.GaussianBlur(mask, (5, 5), 0)
    ret, th = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("mask", th)
    cv2.waitKey(800)
    cv2.destroyAllWindows()
    # cv2.line(img,(0,primTercShape[1]),(x,primTercShape[1]),(0,255,0),thickness=1)
    # cv2.line(img,(0,secTercShape[1]),(x,secTercShape[1]),(0,255,0),thickness=1)
    imgContours, npaContours, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    lenContours = len(npaContours)
    validContours = []
    imaa = img.copy()


    for i in range(0, lenContours):
        cnt = npaContours[i]
        ord = hierarchy[0][i][2]
        x, y, w, h = cv2.boundingRect(cnt)
        ima = img[y:y + h, x:x + w]
        if ((w * h) > 700 and (w * h) <= 3000):
            if (1):  # y > primTercShape[1]):
                ptoX, ptoY = x + w, y + h
                if (1):
                    if (1):  # ord != -1):
                        # print(hierarchy[0][i])
                        # print(" --- punto X,Y ", (x, y), " --- Punto X+W , Y+H --- ", (x + w, y + h))
                        cv2.imshow("ima", ima)
                        cv2.waitKey(800)
                        cv2.destroyWindow("ima")
                        # if(y>=60 and (w*h)>2000):
                        # print(w*h)
                        # print(x,y,w,h)
                        validContours.append([x, y, w, h])
                        # else:continue

    validContours2 = sorted(validContours, key=itemgetter(0))
    cnt = 0
    for x, y, w, h in validContours2:
        if (len(buffer) <= 0):
            break
        cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 126), 3)
        testImage = gray[y:y + h, x:x + w]
        umbral = mask[y:y + h, x:x + w]
        cv2.imshow("imagen", img2)
        cv2.imshow("char", testImage)
        cv2.imshow("umbral", umbral)
        cv2.waitKey(800)
        cv2.destroyAllWindows()
        testImage = umbral
        caracterTest = cv2.resize(testImage, (10, 10), None, 0, 0, cv2.INTER_NEAREST)
        caracterLine = np.asarray(caracterTest.ravel())
        caracterLine = caracterLine.reshape(1, -1)
        # res = classifier.predict(caracterLine)
        res = classifierKNN.predict(caracterLine)
        print("el resultado es", res)
        try:
            index = buffer.index(res)
        except:
            index = -1
        if (index != -1):
            cnt += 1
            buffer.remove(res)
        # cv2.imshow("char",testImage)
        # cv2.waitKey(1000)
        # cv2.destroyAllWindows()
        print(x, y, w, h)
    leng = len(buffer)
    if (leng > 0):  # not all characters detected
        if (cnt >= 3):  # almost all detected
            return "stop"
    else:
        return "stop"


def searchDigits(image, caracterList, labelList):
    print("TO-DO")
    raise Exception("Function TO-DO")
    recognition = []
    classifier = GaussianNB()
    # classifierKNN = KNeighborsClassifier(n_neighbors=1)
    # classifierKNN.fit(caracterList, labelList)
    classifier.fit(caracterList, labelList)
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
    mask = graph.getBinaryInvMask(prueba)
    # mask=cv2.bitwise_not(mask)
    cv2.imshow("mask", mask)
    cv2.waitKey(800)
    cv2.destroyAllWindows()
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    # imgBlurred = cv2.GaussianBlur(gray, (5,5), 0)
    imgBlurred = cv2.GaussianBlur(mask, (5, 5), 0)
    ret, th = cv2.threshold(imgBlurred, 127, 255, cv2.THRESH_BINARY_INV)
    # cv2.line(img,(0,primTercShape[1]),(x,primTercShape[1]),(0,255,0),thickness=1)
    # cv2.line(img,(0,secTercShape[1]),(x,secTercShape[1]),(0,255,0),thickness=1)
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
        ima = img[y:y + h, x:x + w]
        if ((w * h) > 700 and (w * h) <= 2500):
            if (y > primTercShape[1]):
                ptoX, ptoY = x + w, y + h
                if (1):
                    if (1):  # ord != -1):
                        print(hierarchy[0][i])
                        print(" --- punto X,Y ", (x, y), " --- Punto X+W , Y+H --- ", (x + w, y + h))
                        cv2.imshow("ima", ima)
                        cv2.waitKey(800)
                        cv2.destroyWindow("ima")
                        # if(y>=60 and (w*h)>2000):
                        # print(w*h)
                        # print(x,y,w,h)
                        validContours.append([x, y, w, h])
                        # else:continue

    validContours2 = sorted(validContours, key=itemgetter(0))
    cnt = 0
    for x, y, w, h in validContours2:
        cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 126), 3)
        testImage = gray[y:y + h, x:x + w]
        umbral = mask[y:y + h, x:x + w]
        cv2.imshow("imagen", img2)
        cv2.imshow("char", testImage)
        cv2.imshow("umbral", umbral)
        cv2.waitKey(800)
        cv2.destroyAllWindows()
        testImage = umbral
        caracterTest = cv2.resize(testImage, (10, 10), None, 0, 0, cv2.INTER_NEAREST)
        caracterLine = np.asarray(caracterTest.ravel())
        caracterLine = caracterLine.reshape(1, -1)
        res = classifier.predict(caracterLine)
        recognition.append(res)
        # res = classifierKNN.predict(caracterLine)
        print("el resultado es", res)


#
# caracterMatrix, labelMatrix = entrenarOCR()
# image = cv2.imread("./auxiliar_images/stopsign2.jpg")
image = cv2.imread("./Training_Images/Training/Images_Sign_Detection_Benchmark\\14\\00000.ppm")
image2 = cv2.resize(image.copy(), (200, 200))
res = trainAndTest(image2, overwrite=False)
# resultado = searchStop(image2, caracterMatrix, labelMatrix)
# print(resultado)
