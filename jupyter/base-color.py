import cv2
import numpy as np

def cv_show(name,img):
    # 调整窗口大小
    cv2.namedWindow(name, 0)   # 0可调大小，注意：窗口名必须imshow里面的一窗口名一直
    cv2.resizeWindow(name, 1600, 900)   # 设置长和宽
    cv2.imshow(name,img)   
    cv2.waitKey (0)  
    cv2.destroyAllWindows()

path1 = r'image/picture1.jpg'
path2 = r'image/picture2.jpg'
raw_image1 = cv2.imread(path1)
raw_image2 = cv2.imread(path2)
cv_show("raw_image1",raw_image1)
cv_show("raw_image2",raw_image2)
hsv_image1 = cv2.cvtColor(raw_image1, cv2.COLOR_BGR2HSV)
hsv_image2 = cv2.cvtColor(raw_image2, cv2.COLOR_BGR2HSV)
cv_show("hsv_image1",hsv_image1)
cv_show("hsv_image2",hsv_image2)

lower_blue = np.array([90, 43,46])
upper_blue = np.array([99, 255, 255])

mask1 = cv2.inRange(hsv_image1, lower_blue, upper_blue)
mask2 = cv2.inRange(hsv_image2, lower_blue, upper_blue)

cv_show("mask1",mask1)
cv_show("mask2",mask2)
res1 = cv2.bitwise_and(raw_image1, raw_image1, mask=mask1)
res2 = cv2.bitwise_and(raw_image2, raw_image2, mask=mask2)
cv_show("res1",res1)
cv_show("res2",res2)

hsv_thres1 = cv2.adaptiveThreshold(cv2.cvtColor(res1, cv2.COLOR_BGR2GRAY), 255,
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 301, 0)
hsv_thres2 = cv2.adaptiveThreshold(cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY), 255,
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 301, 0)

h,w=hsv_thres1.shape
#建立一个与图像尺寸相同的全零数组
npim = np.zeros((h,w), dtype=np.int)
#将图像3个通道相加赋值给空数组
npim[:]=hsv_thres1[:,:]
#统计白色像素个数
print(len(npim[npim==255]))

h,w=hsv_thres2.shape
#建立一个与图像尺寸相同的全零数组
npim = np.zeros((h,w), dtype=np.int)
#将图像3个通道相加赋值给空数组
npim[:]=hsv_thres2[:,:]
#统计白色像素个数
print(len(npim[npim==255]))



#cv_show("hsv_thres1",hsv_thres1)
#cv_show("hsv_thres2",hsv_thres2)


'''
area = 0

height, width = hsv_thres1.shape
for i in range(height):
    for j in range(width):
        if hsv_thres1[i, j] == 255:
            area += 1
print(area)
'''
'''
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


#plt.figure("lena")
#arr=img.flatten()
#n, bins, patches = plt.hist(arr, bins=256, facecolor='green', alpha=0.75)  
#plt.show()

hist1 = cv2.calcHist([hsv_thres2],[0],None,[256],[0,255])
plt.plot(hist1,color='r')
plt.show()
'''