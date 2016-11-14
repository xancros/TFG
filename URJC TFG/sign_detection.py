# Author: Jose Miguel Buenaposada 2015.
# Simple MSER application for traffic sign window proposal generation
import cv2
import numpy as np
import matplotlib as plt
import os

mser = cv2.MSER_create()

test_dir = 'Imagenes Deteccion/test/'
TRAIN_DIR = 'Imagenes Deteccion/train/'
test_ext = ('.jpg', '.ppm')

def test():
        for filename in os.listdir(test_dir):
            if os.path.splitext(filename)[1].lower() in test_ext:
                print ("Test, processing ", filename, "\n");
                full_path = os.path.join(test_dir, filename)
                I = cv2.imread(full_path)
                Icopy = I.copy()
                Igray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

                regions = mser.detectRegions(Igray, None)
                rects = [cv2.boundingRect(p.reshape(-1,1,2)) for p in regions]
                for r in rects:
                   x,y,w,h = r
                   # Simple aspect ratio filtering
                   aratio = float(w)/float(h)
                   if (aratio > 1.2) or (aratio < 0.8):
                       continue
                   cv2.rectangle(Icopy, (x, y), (x + w, y + h), (0, 255, 0), 2)

                cv2.imshow('img', Icopy)
                cv2.waitKey(0)

def train():
    for folder in os.listdir(TRAIN_DIR):
        parcial_path = os.path.join(TRAIN_DIR, folder)
        if(parcial_path.__contains__("Otros")):
            continue
        for subfolder in os.listdir(parcial_path):
            subF = os.path.join(parcial_path, subfolder)
            for filename in os.listdir(subF):
                if os.path.splitext(filename)[1].lower() in test_ext:
                    print("Test, processing ", subF+"\\"+filename, "\n");
                    full_path = os.path.join(subF, filename)
                    I = cv2.imread(full_path)
                    Icopy = I.copy()
                    Igray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

                    regions = mser.detectRegions(Igray, None)
                    rects = [cv2.boundingRect(p.reshape(-1, 1, 2)) for p in regions]
                    for r in rects:
                        x, y, w, h = r
                        # Simple aspect ratio filtering
                        aratio = float(w) / float(h)
                        if (aratio > 1.2) or (aratio < 0.8):
                            continue
                        cv2.rectangle(Icopy, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    cv2.imshow('img', Icopy)
                    cv2.waitKey(0)

train()