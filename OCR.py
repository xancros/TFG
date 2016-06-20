from os import listdir

import cv2
import numpy as np
from sklearn.lda import LDA

pathTrain = "./Training_Images/OCRImages/Train/training_ocr/"
caracterMatrix = []
carIndex = []
clf = LDA()
diccionario = np.chararray((37, 1))
diccionario = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
               '7': 7, '8': 8, '9': 9, 'A': 10, 'B': 11, 'C': 12, 'D': 13,
               'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19, 'K': 20,
               'L': 21, 'M': 22, 'N': 23, 'O': 24, 'P': 25, 'Q': 26, 'R': 27,
               'S': 28, 'T': 29, 'U': 30, 'V': 31, 'W': 32, 'X': 33, 'Y': 34,
               'Z': 35, 'ESP': 36}


# '0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','ESP'


def entrenarOCR():
    print("funcion de entrenamiento del ocr")
    for archivo in listdir(pathTrain):

        prime = archivo.split('_')[0]
        if (prime != 'A' or prime != 'E' or prime != 'I' or prime != 'U'):
            full_path = pathTrain + archivo
            img = cv2.imread(full_path, 0)
            ret, umbral = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(umbral.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            for cnt in contours:
                carIndex.append(diccionario[archivo.split('_')[0]])
                x, y, w, h = cv2.boundingRect(cnt)
                umbralcut = umbral[y:y + h, x:x + w]
                caracter = cv2.resize(umbralcut, (10, 10), None, 0, 0, cv2.INTER_NEAREST)
                caracterLine = np.asarray(caracter).ravel()
                caracterMatrix.append(caracterLine)


def analizarImagen(imagen):
    print("busca texto STOP en la imagen")
    carMatrix = np.vstack(caracterMatrix)
    matr = []
    x2 = 0
    x3 = 0
    y3 = 0
    y2 = 0
    E = np.array(carIndex)
    clf.fit.transform(carMatrix, carIndex)
    CR = clf.transform(carMatrix)
    CR = CR.astype(np.float32, copy=True)
    # Bayes = cv2.NormalBayesClassifier(CR,responses=E)
    Knear = cv2.KNearest(CR, E, max_k=50)
    # em = cv2.EM()
    # em.train(CR,labels=E)
    img = imagen.copy()
    imgBN = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgTH = cv2.adaptiveThreshold(imgBN, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    contours, hierarchy = cv2.findContours(imgTH.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    cntOrd = []
    parents = []
    for i in range(0, contours.__len__()):
        # x,y,w,h=cv2.boundingRect(cnt)
        # if(y>=y2 and y<y3):
        #     if(x>=x2 and x<x3):
        if (hierarchy[0][i][2] != -1):
            x, y, w, h = cv2.boundingRect(contours[i])
            if (h > (y3 - y2) / 2 and w > (x3 - x2) / 2):
                parents.append(i)
    for padre in parents:
        son = hierarchy[0][padre][2]
        indexes = [son]
        while (hierarchy[0][son][0] != -1):
            son = hierarchy[0][son][0]
            indexes.append(son)
        for i in indexes:
            # if (contours[i].size>50):
            x, y, w, h = cv2.boundingRect(contours[i])
            if (contours[i].size < imgTH.size / 20):
                # print contours[i].size
                x, y, w, h = cv2.boundingRect(contours[i])
                if (h > (y3 - y2) / 2.5 and w > (x3 - x2) / 26):
                    if (w < (x3 - x2) / 7):
                        cv2.rectangle(imagen, (x + x2, y + y2), (x + w + x2, y + h + y2), (255, 0, 255))
                        cntOrd.append((x, y, w, h))

    dtype = [('x', np.int32), ('y', np.int32), ('width', np.int32), ('height', np.int32)]
    rectangulos = np.array(cntOrd, dtype=dtype)
    rectangulos = np.sort(rectangulos, order='x')

    for rect in rectangulos:
        letra = imgTH[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
        caracter = cv2.resize(letra, (10, 10), None, 0, 0, cv2.INTER_NEAREST)
        letralinea = np.asarray(caracter).ravel()
        letralinea = clf.transform(letralinea)
        letralinea = letralinea.astype(np.float32)
        # retval, results = Bayes.predict(np.float32(letralinea))
        retval, results, neighbors, dists = Knear.find_nearest(np.float32(letralinea), 10)
        # retval,probs = em.predict(np.float32(letralinea))
        # print retval
        for key, value in diccionario.iteritems():
            if value == int(retval):
                # print key
                if value != 36:
                    matr.append(key)
                    cv2.putText(imagen, key, (rect[0] + x2, rect[1] - 20 + y2), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), lineType=cv2.CV_AA)
                break
                # print retval

    # print results

    indice = 0
    matrString = []
    if (matr.__len__() != 0):
        # print matr
        if (not matr[0].isdigit()):
            for indice in range(0, matr.__len__()):
                if (indice == 8):
                    # print matrString
                    matrString.append(" ")
                    # print fileName

                matrString.append(matr[indice])
        else:
            for indice in range(0, matr.__len__()):

                if (indice == 7):
                    # print matrString
                    matrString.append(" ")
                    # print fileName

                matrString.append(matr[indice])

        print(matrString)
        # self.listaMatriculas.append(matrString)
    #



    cv2.imshow("imagen", imagen)
    cv2.waitKey()
    cv2.destroyAllWindows()


entrenarOCR()
