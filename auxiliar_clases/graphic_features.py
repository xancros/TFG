import cv2
import numpy as np
from matplotlib import pyplot as plt

from auxiliar_clases import mathFunctions as Vect


def drawPoints(image, listIndex, list, ShowImage=True):
    for index in listIndex:
        pt = (list[index][0], list[index][1])
        cv2.circle(image, pt, 3, (255, 0, 0), -1)
    cv2.imshow("Points in list", image)


def pruebaCirculo(im, imageName):
    if (im is None):
        bgr_img = cv2.imread('./test2.jpg')  # read as it is
    elif (isinstance(im, str)):
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
    imagen, img = limpiarImagen(bgr_img, "circulo")
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
        print(colored("NO hay circulos", 'red'))


def detectarCirculo(imagen):
    print("detectar circulo")
    # image = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)
    imBN = cv2.imread("./test2.jpg", cv2.IMREAD_GRAYSCALE)
    (thresh, im_bw) = cv2.threshold(imBN, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # cv2.imshow("bnn",im_bw)
    # cv2.imshow("bn",imBN)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    circles = cv2.HoughCircles(im_bw, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(imBN, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(imBN, (i[0], i[1]), 2, (0, 0, 255), 3)
    cv2.imshow('detected circles', imBN)
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
