from cmath import pi
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

def nearest(img, scaleW, scaleH, rotateAngle):
    rotateAngle = rotateAngle * pi / 180
    h = img.shape[0]
    w = img.shape[1]
    
    x0 = w // 2
    y0 = h // 2

    sine = math.sin(rotateAngle)
    cosine = math.cos(rotateAngle)
    out = np.full((256,256), 255, dtype = np.uint8)

    for px in range(256):
        for py in range(256):
            wantedX = x0 + int((px-x0)/scaleW)
            wantedY = y0 + int((py-y0)/scaleH)
            if wantedX not in range(0,255) or wantedY not in range(0,255):
                out[px][py] = 255
            else:
                out[px][py] = img[wantedX][wantedY]

    temp = out
    out = np.full((256,256), 255, dtype = np.uint8)
    for px in range(256):
        for py in range(256):
            wantedX = x0 + int((px-x0)*cosine + (py-y0)*sine)
            wantedY = y0 + int((px-x0)*-sine + (py-y0)*cosine)
            if wantedX not in range(0,255) or wantedY not in range(0,255):
                out[px][py] = 255
            else:
                out[px][py] = temp[wantedX][wantedY]
    return out

def bilinear(img, scaleW, scaleH, rotateAngle):
    rotateAngle = rotateAngle * pi / 180
    h = img.shape[0]
    w = img.shape[1]
    
    x0 = w // 2
    y0 = h // 2

    sine = math.sin(rotateAngle)
    cosine = math.cos(rotateAngle)
    out = np.full((256,256), 255, dtype = np.uint8)

    for px in range(256):
        for py in range(256):
            wantedX = x0 + (px-x0)/scaleW
            wantedY = y0 + (py-y0)/scaleH
            if not(0<=wantedX<=255 and 0<=wantedY<=255):
                out[px][py] = 255
            else:
                cX = math.ceil(wantedX)
                cY = math.ceil(wantedY)
                fX = math.floor(wantedX)
                fY = math.floor(wantedY)
                out[px][py] = ((1-wantedX+fX) * (1-wantedY+fY) * img[fX][fY] 
                + (1-wantedX+fX) * (1-cY+wantedY) * img[fX][cY] 
                + (1-cX+wantedX) * (1-cY+wantedY) * img[cX][cY]
                + (1-cX+wantedX) * (1-wantedY+fY) * img[cX][fY])
    temp = out
    out = np.full((256,256), 255, dtype = np.uint8)
    for px in range(256):
        for py in range(256):
            wantedX = x0 + (px-x0)*cosine + (py-y0)*sine
            wantedY = y0 + (px-x0)*-sine + (py-y0)*cosine
            if not(0<=wantedX<=255 and 0<=wantedY<=255):
                out[px][py] = 255
            else:
                cX = math.ceil(wantedX)
                cY = math.ceil(wantedY)
                fX = math.floor(wantedX)
                fY = math.floor(wantedY)
                out[px][py] = ((1-wantedX+fX) * (1-wantedY+fY) * temp[fX][fY] 
                + (1-wantedX+fX) * (1-cY+wantedY) * temp[fX][cY] 
                + (1-cX+wantedX) * (1-cY+wantedY) * temp[cX][cY]
                + (1-cX+wantedX) * (1-wantedY+fY) * temp[cX][fY]) 
    return out

def bicubic(img, scaleW, scaleH, rotateAngle):
    rotateAngle = rotateAngle * pi / 180
    h = img.shape[0]
    w = img.shape[1]
    
    x0 = w // 2
    y0 = h // 2

    sine = math.sin(rotateAngle)
    cosine = math.cos(rotateAngle)
    out = np.full((256,256), 255, )

    for px in range(256):
        for py in range(256):
            wantedX = x0 + (px-x0)/scaleW
            wantedY = y0 + (py-y0)/scaleH
            cX = math.ceil(wantedX)
            cY = math.ceil(wantedY)
            fX = math.floor(wantedX)
            fY = math.floor(wantedY)
            a1 = 255 if ((fX - 1) not in range(0,255) or (fY - 1) not in range(0,255)) else img[fX-1][fY-1]
            a2 = 255 if ((fX - 1) not in range(0,255) or (fY) not in range(0,255)) else img[fX-1][fY]
            a3 = 255 if ((fX - 1) not in range(0,255) or (cY) not in range(0,255)) else img[fX-1][cY]
            a4 = 255 if ((fX - 1) not in range(0,255) or (cY + 1) not in range(0,255)) else img[fX-1][cY+1]
            a5 = 255 if ((fX) not in range(0,255) or (fY - 1) not in range(0,255)) else img[fX][fY-1]
            a6 = 255 if ((fX) not in range(0,255) or (fY) not in range(0,255)) else img[fX][fY]
            a7 = 255 if ((fX) not in range(0,255) or (cY) not in range(0,255)) else img[fX][cY]
            a8 = 255 if ((fX) not in range(0,255) or (cY + 1) not in range(0,255)) else img[fX][cY+1]
            a9 = 255 if ((cX) not in range(0,255) or (fY - 1) not in range(0,255)) else img[cX][fY-1]
            a10 = 255 if ((cX) not in range(0,255) or (fY) not in range(0,255)) else img[cX][fY]
            a11 = 255 if ((cX) not in range(0,255) or (cY) not in range(0,255)) else img[cX][cY]
            a12 = 255 if ((cX) not in range(0,255) or (cY + 1) not in range(0,255)) else img[cX][cY+1]
            a13 = 255 if ((cX + 1) not in range(0,255) or (fY - 1) not in range(0,255)) else img[cX+1][fY-1]
            a14 = 255 if ((cX + 1) not in range(0,255) or (fY) not in range(0,255)) else img[cX+1][fY]
            a15 = 255 if ((cX + 1) not in range(0,255) or (cY) not in range(0,255)) else img[cX+1][cY]
            a16 = 255 if ((cX + 1) not in range(0,255) or (cY + 1) not in range(0,255)) else img[cX+1][cY+1]
            val = (a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15 + a16)
            out[px][py] = 255 if val > 255 else val

    temp = out
    out = np.full((256,256), 255)
    for px in range(256):
        for py in range(256):
            wantedX = x0 + (px-x0)*cosine + (py-y0)*sine
            wantedY = y0 + (px-x0)*-sine + (py-y0)*cosine
            #print(wantedX, wantedY)
            cX = math.ceil(wantedX)
            cY = math.ceil(wantedY)
            fX = math.floor(wantedX)
            fY = math.floor(wantedY)
            a1 = 255 if ((fX - 1) not in range(0,255) or (fY - 1) not in range(0,255)) else temp[fX-1][fY-1]
            a2 = 255 if ((fX - 1) not in range(0,255) or (fY) not in range(0,255)) else temp[fX-1][fY]
            a3 = 255 if ((fX - 1) not in range(0,255) or (cY) not in range(0,255)) else temp[fX-1][cY]
            a4 = 255 if ((fX - 1) not in range(0,255) or (cY + 1) not in range(0,255)) else temp[fX-1][cY+1]
            a5 = 255 if ((fX) not in range(0,255) or (fY - 1) not in range(0,255)) else temp[fX][fY-1]
            a6 = 255 if ((fX) not in range(0,255) or (fY) not in range(0,255)) else temp[fX][fY]
            a7 = 255 if ((fX) not in range(0,255) or (cY) not in range(0,255)) else temp[fX][cY]
            a8 = 255 if ((fX) not in range(0,255) or (cY + 1) not in range(0,255)) else temp[fX][cY+1]
            a9 = 255 if ((cX) not in range(0,255) or (fY - 1) not in range(0,255)) else temp[cX][fY-1]
            a10 = 255 if ((cX) not in range(0,255) or (fY) not in range(0,255)) else temp[cX][fY]
            a11 = 255 if ((cX) not in range(0,255) or (cY) not in range(0,255)) else temp[cX][cY]
            a12 = 255 if ((cX) not in range(0,255) or (cY + 1) not in range(0,255)) else temp[cX][cY+1]
            a13 = 255 if ((cX + 1) not in range(0,255) or (fY - 1) not in range(0,255)) else temp[cX+1][fY-1]
            a14 = 255 if ((cX + 1) not in range(0,255) or (fY) not in range(0,255)) else temp[cX+1][fY]
            a15 = 255 if ((cX + 1) not in range(0,255) or (cY) not in range(0,255)) else temp[cX+1][cY]
            a16 = 255 if ((cX + 1) not in range(0,255) or (cY + 1) not in range(0,255)) else temp[cX+1][cY+1]
            val = (a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15 + a16)
            out[px][py] = 255 if val > 255 else val
    return out.astype(np.uint8)

img = cv2.imread("./images/T.png", cv2.IMREAD_GRAYSCALE)

n = nearest(img, 0.7, 0.7, -15)  # streching scale of the image w h
bl = bilinear(img, 0.75, 0.75, -15)
bc = bicubic(img, 0.7, 0.7, -15)
cv2.imshow('nearest',n)
cv2.imshow('bilinear',bl)
cv2.imshow('bicubic',bc)
#cv2.imwrite('./images/nearest.png', n)
#cv2.imwrite('./images/bilinear.png', bl)
#cv2.imwrite('./images/bicubic.png', bc)
cv2.waitKey(0)
    