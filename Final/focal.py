import numpy as np
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt


def paint(p, h, w):
    color = np.zeros((h, w, 3), np.uint8)
    n = np.max(p) + 1
    i = 255 // n
    for x in range(w):
        for y in range(h):
            color[y][x] = [i*(p[y][x]+1), i*(p[y][x]+1), 0]
    return color


def pick(sum, h, w):
    p = np.zeros((h, w), np.uint8) #record as 3 number for different focus
    for x in range(w):
        for y in range(h):
            p[y][x] = np.where(sum[:,y,x] == max(sum[:,y,x]))[0][0]
    return p


def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and param[0][y][x] != param[2]:
        # first image(goal) weight has to be finally 1, second image(start) has to be 0
        # inserted images number * waitkey time = total transform time (if ignore processing time)
        for i in range(1,11):
            mid = cv2.addWeighted(param[1][param[0][y][x]], 0.1*i, param[1][param[2]], 1-0.1*i, 0)
            cv2.waitKey(40) #ms
            cv2.imshow('image', mid)
        print(param[2], 'to', param[0][y][x])
        param[2] = param[0][y][x]


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
    return lap_sum


def adjust():
    path = './images/image_toss_here/'
    files = [f for f in sorted(os.listdir(path)) if not f.startswith('.')]
    print(files)
    if files == []:
        print('No readable file')
        return False

    # Every image adjust to a best match with image1
    for f in files:
        
        
        img = Image.open(path+f)

        # If read image1, just copy it as the adjusted version and record some information
        if f == files[0]:
            img = np.array(img)
            result = img[:, :, [2, 1, 0]]
            cv2.imwrite('./images/adjust/adjust_'+f, result)
            golden = img
            img_size = golden.shape
            continue
        
        err_final = np.inf
        i_final = 0

        for j in range(1,100):
            if j%10 == 0:
                print(j,'%')
            i = 1+j*0.001
            new_img = img.resize((int(img_size[1]*i), int(img_size[0]*i)))
            new_img = np.array(new_img)
            size_new = new_img.shape

            # bias range of the image center
            if int((size_new[0]-img_size[0])/2) >= 10:
                y_bias = 10
            else:
                y_bias = int((size_new[0]-img_size[0])/2)
            
            if int((size_new[1]-img_size[1])/2) >= 10:
                x_bias =  10
            else:
                x_bias = int((size_new[1]-img_size[1])/2)


            x_record = 0
            y_record = 0

            # Try each bias to find the min err of this resize ratio
            for y_ in range(-y_bias, y_bias+1):
                for x_ in range(-x_bias, x_bias+1):
                    crop_img = new_img[int((size_new[0]-img_size[0])/2)+y_:int((size_new[0]-img_size[0])/2)+img_size[0]+y_,
                                        int((size_new[1]-img_size[1])/2)+x_:int((size_new[1]-img_size[1])/2)+img_size[1]+x_]
            
                    err = np.sum((crop_img-golden)*(crop_img-golden))
                    if(err<err_final):
                        i_final = i
                        err_final =err
                        x_record = x_
                        y_record = y_
                        print('\nAdjusting ',f)
                        print('Resize ratio:', i_final)
                        print('Central bias:')
                        print('  vertical',y_record,'\n  horizontal', x_record)
                        print('Total err:',err_final)

            # Overdo
            if i - i_final > 0.01:
                print('Break')
                break
        print(i_final,err_final)
        print(y_record, x_record)
            
        new_img = img.resize((int(img_size[1]*i_final), int(img_size[0]*i_final)))
        new_img = np.array(new_img)
        size_new = new_img.shape
        crop_img = new_img[int((size_new[0]-img_size[0])/2)+y_record:int((size_new[0]-img_size[0])/2)+img_size[0]+y_record,
                            int((size_new[1]-img_size[1])/2)+x_record:int((size_new[1]-img_size[1])/2)+img_size[1]+x_record]

        result = crop_img[:, :, [2, 1, 0]]
        cv2.imwrite('./images/adjust/adjust_'+f,result)
    
    return True


def refocus():

    func = input('Please choose a function:\n1.Load raw image from "image_toss_here"\n2.Load adjusted image from "adjust"\n3.Load summation numpy array from "npy"\n')

    if func not in ['1','2','3']:
        print('Error input')
        return
    
    flag = True
    # If 3 chosen, run from here
    if func == '1':
        flag = adjust()
        if not flag:
            return

    
    
# Read images into a 3d array (rgb 4d), [No. of image, y, x, (rgb)]
    path = './images/adjust/'
    files = [f for f in sorted(os.listdir(path)) if not f.startswith('.')]
    print(files)
    n = len(files)
    
    if files == []:
        print('No readable file.')
        return
    for f in range(n):
        if f == 0:
            image = cv2.imread(path + files[f], 0)
            h = image.shape[0]
            w = image.shape[1]
            print('Declaring images array')
            gray = np.zeros((n, h, w), np.uint8)
            rgb = np.zeros((n, h, w, 3), np.uint8)
        image = cv2.imread(path + files[f],0)
        gray[f,:,:] = image
        image = cv2.imread(path + files[f])
        rgb[f,:,:,:] = image

    # If 2 chosen, run from here
    if func in ['1','2']:
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
        sum = np.zeros((n, h, w), np.uint32)
        for f in range(n):
            print('processing ',files[f])
            sum[f,:,:] = sum_lap(padded[f,:,:], h, w)
        print('Saving numpy array')
        np.save('./npy/total', sum)
    

    # If 3 chosen, run from here
    print('Loading saved numpy array')
    sum = np.load('./npy/total.npy')

    h = sum.shape[1]
    w = sum.shape[2]

#  Construct a function array p {selected pixel} -> {image to show}
    p = pick(sum, h, w)

# A visualization of detected focus with p
    r = paint(p, h, w)
    

# Showing and setting mouse event  
    param = [p, rgb, 0]
    print('\nDone! Please check the pop-up windows.')
    print('You can click on where you want the focus at.')
    print('Press any key in windows to quit.')
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