import cv2
import numpy as np
from matplotlib import pyplot as plt

Img = cv2.imread('Test.jpg', cv2.IMREAD_GRAYSCALE)

#Filter
SobelX = cv2.Sobel(Img, cv2.CV_64F, 1, 0, ksize=5)  # تشخیص لبه‌ها در جهت افقی
SobelY = cv2.Sobel(Img, cv2.CV_64F, 0, 1, ksize=5)  # تشخیص لبه‌ها در جهت عمودی
Sobel = cv2.magnitude(SobelX, SobelY)

#Show
plt.figure(figsize=(12, 8))
plt.subplot(1, 3, 1)
plt.imshow(Img, cmap='gray')

plt.subplot(1, 3, 2)
plt.title('X')
plt.imshow(SobelX, cmap='gray')
plt.subplot(1, 3, 3)
plt.title('Y')
plt.imshow(SobelY, cmap='gray')

plt.figure(figsize=(6, 6))
plt.title('Sobel')
plt.imshow(Sobel, cmap='gray')
plt.show()
