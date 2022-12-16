import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

                # 1/3   , 1/3
def bilinear(img, scaleW, scaleH):
    h = img.shape[0]
    w = img.shape[1]

    h = math.floor(h*scaleH)
    w = math.floor(w*scaleW)

    out = np.zeros((h,w,3), dtype = np.uint8)
    for x in range(w):
        for y in range(h):
            out[y][x] = img[math.floor(y/scaleH)][math.floor(x/scaleW)]
    return out

if __name__ == '__main__':
    img = cv2.imread('./images/555.jpeg')
    img = bilinear(img,1/3, 1/3)
    cv2.imwrite('./555com.jpeg', img)