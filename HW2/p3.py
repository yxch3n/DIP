import numpy as np
import cv2
import matplotlib.pyplot as plt

def reverse10(filter):
    size = filter.shape
    for u in range(size[1]):
        for v in range(size[0]):
            if filter[v,u] == 1.0:
                filter[v, u] = 0.0
            else:
                filter[v, u] = 1.0
    return filter

def notch_reject_filter(removal_type, shape, radius=9, u_center=0, v_center=0, x=0):
    P, Q = shape
    cQ = Q/2
    cP = P/2
    # Initialize filter with zeros
    H = np.zeros((P, Q))
    # Traverse through filter
    if removal_type == 0:
        for u in range(0, Q):
            for v in range(0, P):
                # Get euclidean distance from point D(u,v) to the center
                D_uv = np.sqrt((v - cP + v_center) ** 2 + (u - cQ + u_center) ** 2)
                D_muv = np.sqrt((v - cP - v_center) ** 2 + (u - cQ - u_center) ** 2)

                if D_uv <= radius or D_muv <= radius:
                    H[v, u] = 0.0
                else:
                    H[v, u] = 1.0
        return H
    else:
        u1 = radius
        v1 = u_center
        u2 = v_center
        v2 = x
        for u in range(0, Q):
            for v in range(0, P):
                if u1<=u<=u2 and v1<=v<=v2:
                    H[v, u] = 0.0
                else:
                    H[v, u] = 1.0
        return H

def tttmean(img):
    img_size = img.shape
    m = np.zeros(img_size)
    for x in range(img_size[0]):
        for y in range(img_size[1]):
            x1 = x - 1
            x2 = x
            x3 = x + 1
            y1 = y - 1
            y2 = y
            y3 = y + 1
            if x == 0:
                x1 = x
            if x == img_size[0] - 1:
                x3 = x
            if y == 0:
                y1 = y
            if y == img_size[1] - 1:
                y3 = y
            m[x][y] = (img[x1][y1] + 
                img[x1][y2] + 
                img[x1][y3] + 
                img[x2][y1] + 
                img[x2][y2] + 
                img[x2][y3] + 
                img[x3][y1] +
                img[x3][y2] +
                img[x3][y3])/9
    return m


if __name__ == "__main__":
    img = cv2.imread("./images/Martian terrain.tif", cv2.IMREAD_GRAYSCALE)
    img_size = img.shape
    print(img_size)

    F = np.fft.fft2(img)
    F = np.fft.fftshift(F)
    spectrum = 2 * np.log(np.abs(F))

    # Circle rejection
    reject = notch_reject_filter(0,img_size, 3, 43, -21)
    reject = reject * notch_reject_filter(0,img_size, 3, 40, -60)
    reject = reject * notch_reject_filter(0,img_size, 3, 55, 24)
    reject = reject * notch_reject_filter(0,img_size, 3, 56, 63)
    reject = reject * notch_reject_filter(0,img_size, 3, 86, -40)
    reject = reject * notch_reject_filter(0,img_size, 3, 89, -18)
    reject = reject * notch_reject_filter(0,img_size, 3, 96, 3)
    reject = reject * notch_reject_filter(0,img_size, 3, 102, 26)
    reject = reject * notch_reject_filter(0,img_size, 3, 124, -123)
    reject = reject * notch_reject_filter(0,img_size, 3, 127, 118)
    # Rectengle rejection
    reject = reject * notch_reject_filter(1,img_size, 122, 0, 125, 110)
    reject = reject * notch_reject_filter(1,img_size, 134, 0, 138, 96)
    reject = reject * notch_reject_filter(1,img_size, 145, 0, 149, 95)
    reject = reject * notch_reject_filter(1,img_size, 125,178, 129, 264)
    reject = reject * notch_reject_filter(1,img_size, 136, 173, 139, 264)
    reject = reject * notch_reject_filter(1,img_size, 149, 166, 152, 264)
    reject = reject * notch_reject_filter(1,img_size, 0, 132, 105, 134)
    reject = reject * notch_reject_filter(1,img_size, 167, 133, 273, 135)
    
    F_filtered = F * reject
    reject = reverse10(reject)
    naive_noise = F * reject
    F_see = 2 * np.log(np.abs(F_filtered))
    naive_noise_img = np.abs(np.fft.ifft2(np.fft.ifftshift(naive_noise)))
    naive_denoise_img = np.abs(np.fft.ifft2(np.fft.ifftshift(F_filtered)))

    # w(x,y) = (a-b)/(c-d)
    # a = mean(g * n)
    a = naive_denoise_img * naive_noise_img
    print(type(a))
    a = tttmean(a)
    
    # b = mean(g) * mean(n)
    b = tttmean(naive_denoise_img) * tttmean(naive_noise_img)

    # c = mean(n**2)
    c = tttmean(naive_noise_img**2)

    # d = mean(n)**2
    d = tttmean(naive_noise_img)**2

    w = (a-b)/(c-d)
    print(w)

    #cv2.imwrite("./images/terrain_naive_noise.tif", denoise_img.astype(np.uint8))
    optimum_notch_img = naive_denoise_img - w * naive_noise_img 
    cv2.imwrite("./images/optimum_restored.tif", optimum_notch_img.astype(np.uint8))

    plt.subplot(231),plt.imshow(spectrum, cmap = 'gray')
    plt.title('Ori Spectrum'),
    plt.xticks([]), plt.yticks([])
    plt.subplot(232),plt.imshow(F_see, cmap = 'gray')
    plt.title('Filtered Spectrum'),
    plt.xticks([]), plt.yticks([])#np.arange(0,271,30)), plt.yticks(np.arange(0,260,30))
    plt.subplot(233),plt.imshow(naive_noise_img, cmap = 'gray')
    plt.title('Naive noise'),
    plt.xticks([]), plt.yticks([])
    plt.subplot(234),plt.imshow(img, cmap = 'gray')
    plt.title('Original image'),
    plt.xticks([]), plt.yticks([])
    plt.subplot(235),plt.imshow(naive_denoise_img, cmap = 'gray')
    plt.title('Naive restored'),
    plt.xticks([]), plt.yticks([])
    plt.subplot(236),plt.imshow(optimum_notch_img, cmap = 'gray')
    plt.title('Optimum restored'),
    plt.xticks([]), plt.yticks([])
    plt.show()