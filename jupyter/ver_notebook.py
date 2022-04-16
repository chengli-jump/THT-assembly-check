import cv2
# 读取保存的图片 （图片在当前目录下。如果不在当前目录，使用绝对路径。）
image = cv2.imread('jupyter/ver.png') 

import matplotlib.pyplot as plt
# 把图片从BGR格式转换为RGB格式
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# 显示图片
plt.imshow(image)
plt.show()

# 将图片转换为灰色
image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# 显示图片
plt.imshow(image_gray, cmap='gray')
plt.show()

# 将图片通过固定的阈值进行二值化
_, image_gray = cv2.threshold(image_gray, 50, 255, cv2.THRESH_BINARY)

# 显示图片
plt.imshow(image_gray, cmap='gray')
plt.show()

# 保存图片至指定的路径
cv2.imwrite("jupyter/image_gray.png", image_gray)

kernel_for_closing = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
close_gray = cv2.morphologyEx(image_gray, cv2.MORPH_CLOSE, kernel_for_closing, iterations=1)

# 显示图片
plt.imshow(close_gray, cmap='gray')
plt.show()

upper = [close_gray.shape[0], 0]
lower = [0, 0]
for i in range(close_gray.shape[0]):
    for j in range(close_gray.shape[1]):
        if close_gray[i][j] == 0:
            if i <= upper[0]:
                upper = [i, j]
            elif i >= lower[0]:
                lower = [i, j]
            # upper = [min(upper[0], i), max(upper[1], j)]
            # lower = [max(lower[0], i), max(lower[1], j)]

print(upper, lower)

import numpy as np

angle = np.arctan( abs(upper[1] - lower[1]) / abs(upper[0] - lower[0]) )
print(angle)