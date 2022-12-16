import os
import cv2
import numpy as np

def ooo():
    path = './images/yes/'
    files = [f for f in sorted(os.listdir(path)) if not f.startswith('.')]
    print(files)
    if files == None:
        print('No readable file')
        return
    for f in range(len(files)):
        if f == 0:
            print(path + files[f])
            image = cv2.imread(path + files[f], 0)
            h = image.shape[0]
            w = image.shape[1]
            print('declair')
            gray = np.zeros((h, w, len(files)), np.uint8)
            rgb = np.zeros((h, w, 3, len(files)), np.uint8)
            print(gray.shape)
        image = cv2.imread(path + files[f],0)
        gray[:,:,f] = image
        image = cv2.imread(path + files[f])
        rgb[:,:,:,f] = image
    print(gray)
    print(rgb)

    cv2.namedWindow("image")
    cv2.imshow('image',rgb[:,:,:,0])
    while(True):
        try:
            cv2.waitKey(100)
        except Exception:
            cv2.destroyWindow("image")
            break
        
    cv2.waitKey(0)
    cv2.destroyAllWindow()





if __name__ == '__main__':
    ooo()