from cmath import exp
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt

def getTheShadowOut(img):
    h = img.shape[0]
    w = img.shape[1]
    k = 1
    ligma = 128
    out = np.zeros((h,w), dtype = float)
    blurrrr = np.arange(256 * 256, dtype = float).reshape(256, 256)
    for i in range(256):
        for j in range(256):
            blurrrr[i][j] = -((i-128)**2 + (j-128)**2)/(2*ligma**2)
    blurrrr = k * np.exp(blurrrr)
    print(blurrrr[128][128])
    x = blurrrr[128,:]
    y = blurrrr[:,128]
    print(x,y)
    for i in range(1024):
        for j in range(1024):
            acc = 0
            pacc = 0
            for a in range(255):
                if (0 < i-128+a < 1024):# and (0 < j-128+b < 1024):
                    pacc += x[a] * img[i-128+a][j]
                    acc += x[a]
            out[i][j] = np.around(pacc/acc)
        print(i)
    temp = out
    for i in range(1024):
        for j in range(1024):
            acc = 0
            pacc = 0
            for a in range(255):
                if (0 < j-128+a < 1024):
                    pacc += y[a] * temp[i][j-128+a]
                    acc += y[a]
            out[i][j] = np.around(pacc/acc)
    return out.astype(np.uint8)

img = cv2.imread("./images/image_4.tif", cv2.IMREAD_GRAYSCALE)
bl = getTheShadowOut(img)
#bl = cv2.GaussianBlur(img, (255,255), 64)
cv2.imshow('a', bl)
cv2.imwrite('./images/shadow.png', bl)
cv2.waitKey(0)
