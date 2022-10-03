# gamma correction
# histogram equalization

import numpy as np
import cv2
import matplotlib.pyplot as plt

def gamma(img, c=1, g=1):
    out = normal = img / 255
    # decide gamma
    while (out > 0.001).all():
        g += 0.2
        out = normal ** g
        print(out.min())
    normal = out
    # decide c
    while (out< 0.999).all() :
        c += 1
        out = normal * c
        print(out.max())
    print('c = ',c,',','gamma = ',g)
    out = np.around(out * 255).astype(np.uint8)
    #hist, bins = np.histogram(out.ravel(),bins = 255,range = [0,255])
    #print("hist gamma =\n",hist)
    #plt.plot(np.arange(0,255,1), hist)
    #plt.show()
    return out

def histo(img):
    hist, bins = np.histogram(img.ravel(),bins = 255,range = [0,255])
    print("hist=\n",hist)
    #print("bins=\n",bins)
    
    proba = hist/img.size 
    #print("pdf=\n",pdf)
    cumul = proba.cumsum()
    #print("cdf=\n",cdf)
    
    equ_value = np.around(cumul * 255).astype(np.uint8)
    #print("equ_value=\n",equ_value)
    result = equ_value[img]
    h, bins = np.histogram(result.ravel(),bins = 255,range = [0,255])
    print(result)
    #plt.plot(np.arange(0,255,1), hist)
    #plt.show()
    #plt.plot(np.arange(0,255,1), h)
    #plt.show()
    return result


img = cv2.imread("./images/image_2.png", cv2.IMREAD_GRAYSCALE)
# 
#cv2.imshow('original', img)
gam = gamma(img)
cv2.imshow('gamma', gam)
cv2.imwrite('./images/gamma.png', gam)
hist = histo(img)
cv2.imwrite('./images/histo.png', hist)
cv2.imshow('histo',hist)
cv2.waitKey(0)