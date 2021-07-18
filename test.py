
import cv2
#img = cv2.imread("C:\\Users\\Asia\\Desktop\\opencv.png")
img = cv2.imread("C:\\Users\\Asia\\Desktop\\opencv.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200, 3)
cv2.imshow("original", img)
cv2.imshow("canny_edges", edges)
'''''''''
x_edges = cv2.Sobel(gray,-1,1,0,ksize=5)
y_edges = cv2.Sobel(gray,-1,0,1,ksize=5)
cv2.imshow("original", img)
cv2.imshow("xedges", x_edges)
cv2.imshow("yedges", y_edges)
'''''''''''
#blur= cv2.medianBlur (img,5)
#new_img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,2)
#new_img1 = cv2.threshold(img,120,255,cv2.THRESH_BINARY)

#blur = cv2.GaussianBlur (img,(5,5),0)
#gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#img_array = cv2.imread("C:\\Users\\Asia\\Desktop\\opencv.png", cv2.IMREAD_GRAYSCALE)
#x_edges = cv2.Sobel(gray,-1,1,0,ksize=5)
#cv2.imwrite("sobel_edges_x.jpg", x_edges)
#y_edges = cv2.Sobel(gray,-1,0,1,ksize=5)
#cv2.imwrite("sobel_edges_y.jpg", y_edges)
#cv2.imshow("original", img)
#cv2.imshow("adapticethreshold", new_img)
#cv2.imshow("threshold", new_img1[1])
#cv2.imshow("IMREAD_GRAYSCALE", img_array)

cv2.waitKey(0)
cv2.destroyAllWindows()
"""""
import numpy as np
import random
import cv2

def sp_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

image = cv2.imread('C:\\Users\\Asia\\Desktop\\opencv.png',1) # Only for grayscale image
noise_img = sp_noise(image,0.05)
cv2.imwrite('C:\\Users\\Asia\\Desktop\\sp_noise.jpg', noise_img)
"""""