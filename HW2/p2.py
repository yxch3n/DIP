from pickletools import uint8
import numpy as np
import cv2
import matplotlib.pyplot as plt

import cv2


def filter(image, kernel):
    m, n = kernel.shape
    if (m == n):
        y, x = image.shape
        y = y - m + 1
        x = x - m + 1
        new_image = np.zeros((y,x))
        for i in range(y):
            for j in range(x):
                new_image[i][j] = np.sum(image[i:i+m, j:j+m]*kernel)
    return new_image


if __name__ == "__main__":
    ori_eistei = cv2.imread('./images/Einstein.tif', cv2.IMREAD_GRAYSCALE)
    ori_phobos = cv2.imread('./images/phobos.tif', cv2.IMREAD_GRAYSCALE)
    kernel = np.array(
    [[0,-0.25,0],
    [-0.25,1,-0.25],
    [0,-0.25,0]])

#Highpass filter
    hp_ei = filter(ori_eistei, kernel)
    hp_ph = filter(ori_phobos, kernel)
    hp_ei = np.abs(hp_ei)
    hp_ph = np.abs(hp_ph)

    ei_trim = np.zeros((598,488), np.uint8)
    for i in range(1,ori_eistei.shape[0]-1):
        for j in range(1,ori_eistei.shape[1]-1):
            ei_trim[i-1][j-1] = ori_eistei[i][j]

    ph_trim = np.zeros((998,681), np.uint8)
    for i in range(1,ori_phobos.shape[0]-1):
        for j in range(1,ori_phobos.shape[1]-1):
            ph_trim[i-1][j-1] = ori_phobos[i][j]

    tmp = (ei_trim + hp_ei).astype(np.uint8)
    hp_eq_ei = cv2.equalizeHist(tmp)
    tmp = (ph_trim + hp_ph).astype(np.uint8)
    hp_eq_ph = cv2.equalizeHist(tmp)

    plt.subplot(221),plt.imshow(ori_eistei, cmap = 'gray')
    plt.title('original'),
    plt.xticks([]), plt.yticks([])
    plt.subplot(222),plt.imshow(hp_eq_ei, cmap = 'gray')
    plt.title('processed'),
    plt.xticks([]), plt.yticks([])
    plt.subplot(223),plt.imshow(ori_phobos, cmap = 'gray')
    plt.title('original'),
    plt.xticks([]), plt.yticks([])
    plt.subplot(224),plt.imshow(hp_eq_ph, cmap = 'gray')
    plt.title('processed'),
    plt.xticks([]), plt.yticks([])
    plt.show()