from os import listdir
from os import path
from os import remove

import cv2
import numpy as np
from sklearn.naive_bayes import GaussianNB

# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# from sklearn.lda import LDA


pathTrain = "./Training_Images/OCRImages/Train/training_ocr/"
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


def entrenarOCR(overwriteFile=True, myPath="./training_file.npz"):
    print("OCR function training")
    if (not myPath.endswith('.npz')):
        myPath += ".npz"
    if (not overwriteFile):
        caracterMatrix, labelMatrix = load(myPath)
    if ((overwriteFile) or (caracterMatrix is None) or (labelMatrix is None)):
        print("training_file not found or overwrite flag set True, let's train :>")
        if (path.exists(myPath)):
            remove(myPath)
        archivos = listdir(pathTrain)
        caracterMatrix, labelMatrix = tratamiento(0, len(archivos), archivos)

        save(caracterMatrix, labelMatrix, myPath)
    print("Training ended")


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


def esNumero(elem):
    try:
        number = int(elem)
        return number
    except:
        return None

def tratamiento(ini, fin, listFiles):
    for index in range(ini, fin):
        file = listFiles[index]
        prime = file.split('_')[0]
        if (not esNumero(prime) is None):
            continue
        # if (prime != 'A' or prime != 'E' or prime != 'I' or prime != 'U'):
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


entrenarOCR(False)
# image = cv2.imread("./auxiliar_images/stopsign.jpg")
# analizarImagen(image)
