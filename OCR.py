from os import listdir

import cv2
import numpy as np

pathTrain = "./Training_Images/OCRImages/Train/"
caracterMatrix = []

def entrenarOCR(self):
    print("funcion de entrenamiento del ocr")
    for archivo in listdir(pathTrain):
        prime = archivo.split('_')[0]
        if (prime != 'A' or prime != 'E' or prime != 'I' or prime != 'U'):
            img = cv2.imread("training_ocr/%s" % (archivo), 0)
            ret, umbral = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(umbral.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            for cnt in contours:
                self.carIndex.append(self.diccionario[archivo.split('_')[0]])
                x, y, w, h = cv2.boundingRect(cnt)
                umbralcut = umbral[y:y + h, x:x + w]
                caracter = cv2.resize(umbralcut, (10, 10), None, 0, 0, cv2.INTER_NEAREST)
                caracterLine = np.asarray(caracter).ravel()
                caracterMatrix.append(caracterLine)


def analizarImagen(self, imagen):
    print("busca texto STOP en la imagen")
