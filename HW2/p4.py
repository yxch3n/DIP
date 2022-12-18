import numpy as np
import cv2
import matplotlib.pyplot as plt

def pad(img):
    h, w = img.shape
    img_out = np.zeros((2*h, 2*w))
    img_out[:h,:w] = img
    return img_out

def normalize(img):
    img_temp = img.astype(np.float)
    img_min = np.min(img)
    img_max = np.max(img)
    img_new = (img-img_min)/(img_max-img_min)*255.0
    return img_new.astype(np.uint8)

def harmo(img):
    img_size = img.shape
    m = np.zeros(img_size)
    x1 = np.zeros(3, np.uint)
    y1 = np.zeros(3, np.uint)
    for x in range(img_size[0]):
        for y in range(img_size[1]):
            x1[0] = x - 1
            x1[1] = x
            x1[2] = x + 1
            y1[0] = y - 1
            y1[1] = y
            y1[2] = y + 1
            if x == 0:
                x1[0] = x
            if x == img_size[0] - 1:
                x1[2] = x
            if y == 0:
                y1[0] = y
            if y == img_size[1] - 1:
                y1[2] = y
            div = 0
            for i in range(3):
                for j in range(3):
                    sum = img[x1[i]][y1[j]]
                    div += sum
            '''for i in range(3):
                for j in range(3):
                    sum = 1 if img[x1[i]][y1[j]] == 0 else img[x1[i]][y1[j]]
                    div += 1/sum'''
            m[x][y] = div/9
    return m

def football():
    raw_img = cv2.imread('./images/Football_players_degraded.tif', cv2.IMREAD_GRAYSCALE)
    img_size = raw_img.shape
    y, x  = img_size
    # move speed
    a = -7/x
    b = 7/y

    # assume constant
    # interactively try
    snr = 0.005

    # pad [-700 ]
    u, v = np.meshgrid(np.arange(2*x)-x, np.arange(2*y)-y)
    # H  function definition

    H = np.sinc(a*u+b*v) * np.exp(-1j*np.pi*(a*u+b*v))
    H_see = 2 * np.log(np.abs(H))

    raw_spectrum = np.fft.fftshift(np.fft.fft2(pad(raw_img)))
    raw_see = 2 * np.log(np.abs(raw_spectrum))

    # F_hat(restored) = G(motion)/H = (1/H) * G  = W * G
    res_spectrum = np.conj(H) / (np.abs(H)**2 + snr) * raw_spectrum * 5

    res_see = 2 * np.log(np.abs(res_spectrum))
    res_img = normalize(np.abs(np.fft.ifft2(np.fft.ifftshift(res_spectrum)))[:y,:x])
    eq_img = cv2.equalizeHist(res_img)


    cv2.imwrite('./images/restored_football.tif', eq_img.astype(np.uint8))

    plt.subplot(231),plt.imshow(raw_see, cmap = 'gray')
    plt.title('Raw spectrum'),
    plt.xticks([]), plt.yticks([])
    plt.subplot(232),plt.imshow(H_see, cmap = 'gray')
    plt.title('H spectrum'),
    plt.xticks([]), plt.yticks([])
    plt.subplot(233),plt.imshow(raw_img, cmap = 'gray')
    plt.title('Raw img'),
    plt.xticks([]), plt.yticks([])
    plt.subplot(234),plt.imshow(res_see, cmap = 'gray')
    plt.title('Restored spectrum'),
    plt.xticks([]), plt.yticks([])
    plt.subplot(235),plt.imshow(res_img, cmap = 'gray')
    plt.title('restored img'),
    plt.xticks([]), plt.yticks([])
    plt.subplot(236),plt.imshow(eq_img, cmap = 'gray')
    plt.title('Equalized img'),
    plt.xticks([]), plt.yticks([])
    plt.show()

def photographer():
    raw_img = cv2.imread('./images/Photographer_degraded.tif', cv2.IMREAD_GRAYSCALE)
    #raw_img = cv2.GaussianBlur(raw_img, (3,3), 0)
    #harmo_img = harmo(raw_img)
    img_size = raw_img.shape
    y, x  = img_size
    # move speed
    a = -8/x
    b = 9/y

    # assume constant
    # interactively try
    snr = 0.1

    # pad shift axis
    u, v = np.meshgrid(np.arange(2*x)-x, np.arange(2*y)-y)
    
    # H  function definition
    H = np.sinc(a*u+b*v) * np.exp(-1j*np.pi*(a*u+b*v))
    H_see = 2 * np.log(np.abs(H))

    raw_spectrum = np.fft.fftshift(np.fft.fft2(pad(raw_img)))
    #raw_see = 2 * np.log(np.abs(raw_spectrum))

    # F_hat(restored) = G(motion)/H = (1/H) * G  = W * G
    res_spectrum = np.conj(H) / (np.abs(H)**2 + snr) * raw_spectrum * 5

    #res_see = 2 * np.log(np.abs(res_spectrum))
    res_img = np.abs(np.fft.ifft2(np.fft.ifftshift(res_spectrum)))[:y,:x]
    res_img = normalize(res_img)
    #eq_img = cv2.equalizeHist(normalize(res_img))
    cv2.imwrite('./images/restored_photographer.tif', res_img.astype(np.uint8))
    plt.subplot(121),plt.imshow(raw_img, cmap = 'gray')
    plt.title('Raw img'),
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(res_img, cmap = 'gray')
    plt.title('Wiener filtered'),
    plt.xticks([]), plt.yticks([])
    plt.show()

if __name__ == "__main__":
    football()
    photographer()
