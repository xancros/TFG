# Author: Jose Miguel Buenaposada 2015.
# Simple MSER application for traffic sign window proposal generation
import cv2
import numpy as np
import matplotlib as plt
import os

mser = cv2.MSER() 

test_dir = 'ImgsAlumnos/test/'
test_ext = ('.jpg', '.ppm')

for filename in os.listdir(test_dir):
    if os.path.splitext(filename)[1].lower() in test_ext:
        print "Test, processing ", filename, "\n"
        full_path = os.path.join(test_dir, filename)
        I = cv2.imread(full_path)
        Icopy = I.copy()
        Igray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

        regions = mser.detect(Igray)
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
