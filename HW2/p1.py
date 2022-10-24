import numpy as np
import cv2
import matplotlib.pyplot as plt

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


#p = np.zeros((1134,1360), dtype=np.int64)

if __name__ == "__main__":
    img = cv2.imread("./images/keyboard.tif", cv2.IMREAD_GRAYSCALE)
    img_size = img.shape
#a
    F = np.fft.fft2(img)
    F = np.fft.fftshift(F)
    spectrum = 20 * np.log(np.abs(F))

#b
    sobel = np.array((
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]), dtype=np.int64)
    odd_sobel = np.pad(sobel,((1,0),(1,0)), 'constant', constant_values = 0)
    print('Original sobel')
    print(sobel)
    print('Odd-symmetric sobel')
    print(odd_sobel)

#c
    h_odd = np.pad(odd_sobel,((0,img_size[0] -4),(0,img_size[1] -4)), 'constant', constant_values = 0)
    print('sobel odd pad')
    print(h_odd.shape)
    print(h_odd)
    H_ODD = np.fft.fft2(h_odd)
    H_ODD = np.fft.fftshift(H_ODD)
    H_ODD = np.imag(H_ODD)
    spectrum2 = 20 * np.abs(H_ODD)

    filtered2 = F * H_ODD
    filtered2 = np.fft.ifft2(np.fft.ifftshift(filtered2))
    filtered2 = np.abs(filtered2)

#d
    #space = cv2.filter2D(img, -1, odd_sobel)
    space = filter(img, sobel)
    space = np.abs(space)

#e 
    h = np.pad(sobel,((0,img_size[0]-3),(0,img_size[1]-3)), 'constant', constant_values = 0)
    H = np.fft.fft2(h)
    H = np.fft.fftshift(H)
    H = np.imag(H)
    spectrum1 = 20 * np.abs(H)

    filtered = F * H
    filtered = np.fft.ifft2(np.fft.ifftshift(filtered))
    filtered = np.abs(filtered)

    plt.subplot(221),plt.imshow(spectrum, cmap = 'gray')
    plt.title('(a) Spectrum'),
    plt.xticks([]), plt.yticks([])
    plt.subplot(222),plt.imshow(filtered2, cmap = 'gray')
    plt.title('(c) Freq domain with odd'), 
    plt.xticks([]), plt.yticks([])
    plt.subplot(223),plt.imshow(space, cmap = 'gray')
    plt.title('(d) Spatial domain'), 
    plt.xticks([]), plt.yticks([])
    plt.subplot(224),plt.imshow(filtered, cmap = 'gray')
    plt.title('(e) Freq domain without odd'), 
    plt.xticks([]), plt.yticks([])
    plt.show()
