import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


def paint(p, h, w):
    color = np.zeros((h, w, 3), np.uint8)
    draw = [[50,50,0],[100,100,0],[150,150,0],[200,200,0],[255,255,0]]
    for x in range(w):
        for y in range(h):
            color[y][x] = draw[p[y][x]]
    return color


def pick(sum, h, w):
    p = np.zeros((h, w), np.uint8) #record as 3 number for different focus
    for x in range(w):
        for y in range(h):
            p[y][x] = np.where(sum[:,y,x] == max(sum[:,y,x]))[0][0]
    return p


def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and param[0][y][x] != param[2]:
        for i in range(1,11):
            mid = cv2.addWeighted(param[1][param[0][y][x]], 0.1*i, param[1][param[2]], 1-0.1*i, 0)
            cv2.waitKey(40)
            cv2.imshow('image', mid)
        print(param[2], 'to', param[0][y][x])
        param[2] = param[0][y][x]
        #cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,0,0), thickness = 1)
        #cv2.imshow("image", param)
        #cv2.imshow("image", param[param[0][y][x]+1])
        

def pad(img, h, w):
    dx = w//10
    dy = h//10
    pad = np.zeros((h + 2*dy, w + 2*dx), np.uint8)
    pad[dy:dy+h,dx:dx+w] = img
    return pad


def sum_lap(lap, h, w):
    dx = w//10
    dy = h//10
    lap_sum = np.zeros((h, w), np.uint32)
    a = 0
    for x in range(dx, dx+w):
        if (x-dx)%(w//10) == 0:
            print(a,'%')
            a += 10
        for y in range(dy, dy+h):
            lap_sum[y-dy][x-dx] = np.sum(lap[y-dy:y+dy, x-dx:x+dx])
    print(lap_sum)
    return lap_sum

def refocus():

# Read images into a 3d array (rgb 4d), [No. of image, y, x, (rgb)]
    path = './images/yes/'
    files = [f for f in sorted(os.listdir(path)) if not f.startswith('.')]
    print(files)
    n = len(files)
    
    if files == None:
        print('No readable file.')
        return
    for f in range(n):
        if f == 0:
            print(path + files[f])
            image = cv2.imread(path + files[f], 0)
            h = image.shape[0]
            w = image.shape[1]
            print('declare images array')
            gray = np.zeros((n, h, w), np.uint8)
            rgb = np.zeros((n, h, w, 3), np.uint8)
        image = cv2.imread(path + files[f],0)
        gray[f,:,:] = image
        image = cv2.imread(path + files[f])
        rgb[f,:,:,:] = image

# Laplacian filtering
    print('Laplace filtering')

    lap = np.zeros((n, h, w), np.uint8)
    for f in range(n):
        lap[f,:,:] = cv2.Laplacian(gray[f,:,:], -1, ksize=3)
    

# Padding laplacian result for following summation

    print('Padding laplace ')
    dx = w//10
    dy = h//10
    padded = np.zeros((n, h + 2*dy, w + 2*dx), np.uint8)
    for f in range(n):
        padded[f,:,:] = pad(lap[f,:,:], h, w)


# Summing laplacian result and save it yo npy file
# (if saved then load it)

    print('Summing gradient')

    '''sum = np.zeros((n, h, w), np.uint32)
    for f in range(n):
        print('processing image',f)
        sum[f,:,:] = sum_lap(padded[f,:,:], h, w)

    np.save('./npy/total', sum)'''
    sum = np.load('./npy/total.npy')

#  Construct a function array p {selected pixel} -> {image to show}
    p = pick(sum, h, w)

# A visualization of detected focus with p
    r = paint(p, h, w)
    

# Showing and setting mouse event  
    param = [p, rgb, 0]
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN, param)
    cv2.imshow("image", rgb[0,:,:,:])
    cv2.namedWindow('focus')
    cv2.imshow("focus", r)
    

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return

if __name__ == "__main__":
    refocus()