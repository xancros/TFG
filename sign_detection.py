# Author: Jose Miguel Buenaposada 2015.
# Simple MSER application for traffic sign window proposal generation
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import math
from termcolor import colored

mser = cv2.MSER_create()
mserTrain = cv2.MSER_create()
TRAIN_DIR="C:/Users/xancr/Documents/Imgs/Final_Training/Images"
#test_dir = './Final_Test/Images'
test_dir = './TEST_Calle'
test_ext = ('.jpg', '.ppm')
print (cv2.__version__)
gg = 0
listDescriptor = []
listAreas=[]
#Descriptor FLANN
FLANN_INDEX_KDTREE = 3
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(indexParams=index_params,searchParams=search_params)
#####################

#Descriptor BF

bf = cv2.BFMatcher(normType=cv2.NORM_HAMMING)

def contienePunto(puntos,punto):
    if(len(puntos)>0):
        r=puntos[-1]
        x,y,w,h=punto
        xR,yR,wR,hR=r
        if(x==xR and y==yR and w==wR and h==hR):
            return True
    return False
#####################

def getHOGDescriptorVisualImage():
    print("parsing")
    ##

    ##

def train ():
    indice =0
    for folder in os.listdir(TRAIN_DIR):
        parcial_path = os.path.join(TRAIN_DIR, folder)
        for filename in os.listdir(parcial_path):
            if os.path.splitext(filename)[1].lower() in test_ext:
                print("Test, processing ", filename, "\n")
                full_path = os.path.join(parcial_path, filename)
                regions = []
                I = cv2.imread(full_path)
                trainShape = I.shape
                pruebaLineas(None,"otro")
                imageShape = (trainShape[1],trainShape[0])
                Icopy = I.copy()
                Igray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
                mser.setDelta(2)
                mser.setMinArea(50)
                regionsDetected = mser.detectRegions(Igray, None)
                # rects = [cv2.boundingRect(p.reshape(-1,1,2)) for p in regionsDetected]
                rects = []
                areas = []
                puntos = []
                aratio = 0
                areaSuficiente=False
                xS,yS,wS,hS=0,0,0,0
                Icopy2=Icopy.copy()
                for p in regionsDetected:
                    rect = cv2.boundingRect(p.reshape(-1, 1, 2))
                    x, y, w, h = rect
                    A = x
                    B = x+w
                    C = y
                    D = y+h

                    # print ("W/H ==> ",(w/h))
                    # hay que hacer un filtro como el de aspect ratio pero con la aparencia de los puntos(x,y,w,h)
                    if (float(w) / float(h) > 1.2) or (float(w) / float(h) < 0.8):
                        continue
                    if(not contienePunto(rects,rect)):
                        if (w==h):
                            if(w-x>=30 and h-y>=30):
                                print(colored("El area es suficientemente grande",'green'))
                                # cv2.rectangle(Icopy, (x, y), (x + w, y + h), (0, 255, 0), thickness=1)
                                areaSuficiente=True
                                # xS,yS,wS,hS=rect
                                rects.append(rect)
                            else:
                                continue
                                # print(colored("El area seleccionada es la imagen", 'red'))
                                #cv2.rectangle(Icopy, (0, 0), (trainShape[0], trainShape[1]), (0, 255, 0), thickness=1)


                            # print (rect)


                print(colored(full_path, 'green'))

                if(areaSuficiente):
                    # cv2.imshow("area", Icopy)
                    # cv2.waitKey()
                    # cv2.destroyWindow("area")
                    # Icopy = Icopy2.copy()
                    min = trainShape[0]*trainShape[1]
                    for x,y,w,h in rects:
                        area=w*h
                        if(area<min):
                            min=area
                            xS,yS,wS,hS=x,y,w,h
                    nuevaImagen = I[xS:xS + wS, yS:yS + hS]
                    cv2.imshow("original",I)
                    cv2.imshow("nuevaImagen",nuevaImagen)
                    cv2.waitKey()
                    cv2.destroyAllWindows()
                    pruebaCirculo(nuevaImagen,filename)
                else:

                    pruebaCirculo(I,filename)


def lineas(imageName=None):
    if(imageName is None):
        fn = cv2.imread("./ceda.jpg")
    else:
        fn = cv2.imread(imageName)
    # src = cv2.imread(fn)
    src = fn.copy()
    imShape = fn.shape
    bn = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
    dst = cv2.Canny(bn, 50, 200)
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    lines = cv2.HoughLines(dst,1,np.pi/180,200)
    # lines = cv2.HoughLines(dst, 1, np.pi / 180, 1.5, 0.0)
    a, b, c = lines.shape
    # linesConverted = np.zeros(dtype=np.float32,shape=(len(lines),2))
    linesConverted = []
    for i in range(len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0, y0 = a * rho, b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        # pt1 = (abs(int(x0 + imShape[0] * (-b))), int(y0 + imShape[1] * (a)))
        # pt2 = (abs(int(x0 - imShape[1] * (-b))), int(y0 - imShape[0] * (a)))
        punto = np.array([pt1,pt2])
        linesConverted.append(punto)
        cv2.circle(src,pt1,4,(255,0,0),-1)
        cv2.circle(src,pt2,4,(0,255,0),-1)
        # linesConverted[i][0]=pt1
        # linesConverted[i][1]=pt2
        cv2.line(src, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow("mm",src)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
    # cv2.imshow("mm",src)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    acumularPuntosInterseccion(np.asarray(linesConverted),fn)
    cv2.imshow("source", fn)
    # cv2.imshow("detected lines", cdst)
    cv2.waitKey()
def otracosa():
    gg=0
    indice = 0
    for filename in os.listdir(test_dir):
        if os.path.splitext(filename)[1].lower() in test_ext:
            print ("Test, processing ", filename, "\n")
            full_path = os.path.join(test_dir, filename)
            regions=[]
            I = cv2.imread(full_path)
            # cv2.imshow("vn",I)
            # cv2.waitKey()
            # cv2.destroyWindow("vn")
            Icopy = I.copy()
            Igray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
            mser.setDelta(2)
            mser.setMinArea(200)
            regionsDetected = mser.detectRegions(Igray,None)
            #rects = [cv2.boundingRect(p.reshape(-1,1,2)) for p in regionsDetected]
            rects = []
            areas = []
            puntos = []
            aratio = 0
            for p in regionsDetected:
                rect = cv2.boundingRect(p.reshape(-1,1,2))
                x,y,w,h = rect
                #hay que hacer un filtro como el de aspect ratio pero con la aparencia de los puntos(x,y,w,h)

                    # Simple aspect ratio filtering
                if (float(w) / float(h) > 1.2) or (float(w) / float(h) < 0.8):
                    continue
                if(not contienePunto(puntos,rect)):
                    cv2.rectangle(Icopy, (x, y), (x + w, y + h), (0, 255, 0), thickness=1)
                    #nueva2Cut = nueva2[y1:y1 + y2, x1:x1 + x2]
                    nuevaImagen=I[y:y+h,x:x+w]
                    # cv2.imshow("area",nuevaImagen)
                    # cv2.waitKey()
                    # cv2.destroyWindow("area")
                    areas.append(nuevaImagen)
                    puntos.append(rect)
                    regions.append(p)

            cv2.imshow('img', Icopy)
            oldPoint = 0,0
            for area in areas:
                a,b,c,d = puntos[indice]
                newPoint = a,b
                if(oldPoint!=newPoint):
                    cv2.imshow("area",area)
                    cv2.waitKey()
                    cv2.destroyWindow("area")
                    print("punto obtenido en el area: \n", puntos[indice])
                    oldPoint=newPoint
                else:
                    print(newPoint," = ", oldPoint)
                indice+=1
            # cv2.waitKey()
            # cv2.destroyWindow("img")
            indice = 0
            if(gg==5):
                break
            else:
                gg+=1
def limpiarImagen(bgr_img,shapeName):
    if(shapeName=="circulo"):
        if bgr_img.shape[-1] == 3:  # color image
            b, g, r = cv2.split(bgr_img)  # get b,g,r
            rgb_img = cv2.merge([r, g, b])  # switch it to rgb
            gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = bgr_img
        # img = cv2.medianBlur(gray_img, 5)
        img = cv2.GaussianBlur(gray_img, (9, 9), 2, sigmaY=2)
        cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return (rgb_img.copy(),img)
    else:
        if bgr_img.shape[-1] == 3:  # color image
            b, g, r = cv2.split(bgr_img)  # get b,g,r
            rgb_img = cv2.merge([r, g, b])  # switch it to rgb
            gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = bgr_img
        # img = cv2.medianBlur(gray_img, 5)
        img = cv2.GaussianBlur(gray_img, (9, 9), 2, sigmaY=2)
        img= cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,75,10)
        img = cv2.bitwise_not(img)
        return (rgb_img.copy(),img)

def pruebaLineas(im,imageName):
    if(im is None):
        bgr_img = cv2.imread("./rectanguloConFlecha.jpg")
    else:
        bgr_img = im.copy()
    gray = cv2.cvtColor(bgr_img,cv2.COLOR_BGR2GRAY)
    cv2.imshow("mm",gray)
    cv2.waitKey()
    cv2.destroyAllWindows()

    imagen,img = limpiarImagen(bgr_img,"otro")
    rgb_img = imagen.copy()
    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    lineas = cv2.HoughLinesP(img,1, 3.14/180, 80, 100, 10)
    whOld=1000
    lineaAnterior = [0,0,0,0]
    if (lineas is not None):
        for linea in lineas:
            x1,y1,x2,y2 = linea[0]
            pt1 = (x1,y1)
            pt2 = (x2,y2)

            if(1):
                print ("linea: punto1 =  ",pt1," punto 2 =",pt2)

                cv2.line(imagen,(x1,y1),(x2,y2),(0,0,255),2)
                cv2.circle(imagen, (x1, y1), 3, (255, 0, 0), -1)
                cv2.circle(imagen, (x2, y2), 3, (255, 0, 0), -1)
                cv2.imshow("mm",imagen)
                cv2.waitKey(5000)
                cv2.destroyAllWindows()
                whOld=x2/y2
                imagen = rgb_img.copy()
        # plt.figure(imageName)
        # plt.subplot(121), plt.imshow(rgb_img)
        # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        # plt.subplot(122), plt.imshow(imagen)
        # plt.title('Hough Transform'), plt.xticks([]), plt.yticks([])
        # plt.show()
    else:
        print("HOLA")



def pruebaCirculo(im,imageName):
    if(im is None):
        bgr_img = cv2.imread('./test2.jpg')  # read as it is
    elif(isinstance(im,str)):
        bgr_img = cv2.imread(im)
    else:
        bgr_img = im.copy()

       # bgr_img = cv2.imread("C:/Users/xancr/Documents/Imgs/Final_Training/Images/00000/00000_00000.ppm")
    # cv2.imshow("im",bgr_img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # if bgr_img.shape[-1] == 3:  # color image
    #     b, g, r = cv2.split(bgr_img)  # get b,g,r
    #     rgb_img = cv2.merge([r, g, b])  # switch it to rgb
    #     gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    # else:
    #     gray_img = bgr_img
    # imagen = rgb_img.copy()
    # # img = cv2.medianBlur(gray_img, 5)
    # img = cv2.GaussianBlur(gray_img,(9,9),2,sigmaY=2)
    imagen,img = limpiarImagen(bgr_img,"circulo")
    rgb_img = imagen.copy()
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 2, 100,
                               param1=10, param2=10, minRadius=0, maxRadius=0)
    if (circles is not None):
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(imagen, (i[0], i[1]), i[2], (0, 255, 0), 1)
            # draw the center of the circle
            cv2.circle(imagen, (i[0], i[1]), 1, (0, 0, 255), 2)

        # cv2.imshow("im",imagen)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        plt.figure(imageName)
        plt.subplot(121), plt.imshow(rgb_img)
        plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(imagen)
        plt.title('Hough Transform'), plt.xticks([]), plt.yticks([])
        plt.show()

    else:
        print (colored("NO hay circulos",'red'))

def detectarCirculo(imagen):
    print("detectar circulo")
    #image = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)
    imBN=cv2.imread("./test2.jpg",cv2.IMREAD_GRAYSCALE)
    (thresh, im_bw) = cv2.threshold(imBN, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # cv2.imshow("bnn",im_bw)
    # cv2.imshow("bn",imBN)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    circles = cv2.HoughCircles(im_bw,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=0)
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        #draw the outer circle
        cv2.circle(imBN,(i[0],i[1]),i[2],(0,255,0),2)
        #draw the center of the circle
        cv2.circle(imBN,(i[0],i[1]),2,(0,0,255),3)
    cv2.imshow('detected circles',imBN)
    cv2.waitKey()
    cv2.destroyAllWindows()

def usoHOG(img=None):
    if(img==None):
        img = cv2.imread("./test2.jpg",0)
    img2=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imres=cv2.resize(img,(64,128))
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
    hog2=cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
    locat=[]

    salida=hog2.compute(img,(0,0),(0,0),locat)
    ##llamada para obtener imagen

    blocks = (winSize[1]/(blockStride[1])+winSigma)
    cellsBlock=(winSize[0]/(blockStride[0])+winSigma )
    length = blocks * cellsBlock *(nbins)*4
    print (hog2.getDescriptorSize())
    print ("prueba de HOG")
    cv2.imshow("imagen",imres)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    hog = cv2.HOGDescriptor()
    #hog.setSVMDetector(cv2.HOGDESCRIPTOR_DEFAULT_NLEVELS)
    des=[]
    lst=[]


    hist=hog.compute(img2,(0,0),(0,0),lst)
    ow,oh,ch=img.shape

    print (hist.shape)
    hist2 = hog.computeGradient(img)

    # cv2.imshow("algo",hist)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    #
    # print (hist)


def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

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
                cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
                if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in range(4)])
                    if max_cos < 0.1:
                        squares.append(cnt)
    return squares

def perp( a ) :
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

# line segment a given by endpoints a1, a2
# line segment b given by endpoints b1, b2
# return
def seg_intersect(a1,a2, b1,b2) :
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot( dap, db)
    num = np.dot( dap, dp )
    return (num / denom.astype(float))*db + b1

p1 = np.array( [0, 3] )
p2 = np.array( [1, 5] )

p3 = np.array( [0, 5] )
p4 = np.array( [1, 2] )


def det(a, b):
    return a[0] * b[1] - a[1] * b[0]

def line_intersection(line1,line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])  # Typo was here


    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def acumularPuntosInterseccion(lines,im):
    # for (int i = 0; i < lines.size(); i++)
    # {
    # for (int j = i + 1; j < lines.size(); j++)
    # {
    # cv::
    #     Point2f
    # pt = computeIntersectionOfTwoLine(lines[i], lines[j]);
    # if (pt.x >= 0 & & pt.y >= 0 & & pt.x < image.cols & & pt.y < image.rows)
    # {
    # corners.push_back(pt);
    # }
    # }
    # }
    # cv2.imshow("mm",im)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    im2 = im.copy()
    imShape = im.shape
    linesShape = lines.shape
    ruptura = []
    sizeLines = len(lines)
    for i in range(sizeLines):
        for j in range(i+1,sizeLines):
            linea1 = lines[i]
            linea2 = lines[j]
            x00,y00 = linea1[0]
            x01, y01 = linea1[1]
            x10,y10 = linea2[0]
            x11, y11 = linea2[1]
            cv2.line(im2,(int(x00),int(y00)),(int(x01),int(y01)),(0,0,255),1,cv2.LINE_AA)
            cv2.line(im2, (int(x10), int(y10)), (int(x11), int(y11)),(255,0,0) ,1, cv2.LINE_AA)
            # cv2.imshow("mm", im2)
            # cv2.destroyAllWindows()


            point = seg_intersect(lines[i][0],lines[i][1],lines[j][0],lines[j][1])
            if(point[0]>0):
                cv2.circle(im2, (int(point[0]), int(point[1])), 4, (0, 255, 0), -1)
                cv2.imshow("mm", im2)
                cv2.destroyAllWindows()
            if(point[0]>=0 and point[1] >=0 and point[1]<imShape[0] and point[0]<imShape[1]):

                ruptura.append(point)
            im2 = im.copy()
    if(True):
        for pt in ruptura:
            x,y = pt
            cv2.circle(im2,(int(x),int(y)),2,(255,0,0),-1)
            cv2.imshow("mm",im2)
            cv2.waitKey()
            cv2.destroyAllWindows()

def otrosEjemplos():
    img = cv2.imread("./rect2.png")
    squares = find_squares(img)
    cv2.drawContours(img, squares, -1, (0, 255, 0), 3)
    cv2.imshow('squares', img)
    cv2.waitKey()
    cv2.destroyAllWindows()
# otrosEjemplos()
# usoHOG()
# train()
# pruebaLineas(None,None)
lineas("./rectanguloS.jpg")
# print (seg_intersect( p1,p2, p3,p4))
# pruebaCirculo("./ceda.jpg","ceda")