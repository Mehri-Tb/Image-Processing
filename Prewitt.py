import cv2
import numpy as np
from matplotlib import pyplot as plt

Img = cv2.imread('Test.jpg', cv2.IMREAD_GRAYSCALE)

#Filter
PreX = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
PreY = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
PreX_Img = cv2.filter2D(Img, -1, PreX)
PreY_Img = cv2.filter2D(Img, -1, PreY)
Prewitt = cv2.magnitude(PreX_Img.astype(np.float64), PreY_Img.astype(np.float64))

#Show
plt.figure(figsize=(12, 8))

plt.subplot(1, 3, 1)
plt.imshow(Img, cmap='gray')
plt.subplot(1, 3, 2)
plt.title('X')
plt.imshow(PreX_Img, cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Y')
plt.imshow(PreY_Img, cmap='gray')

plt.figure(figsize=(6, 6))
plt.title('Prewitt')
plt.imshow(Prewitt, cmap='gray')
plt.show()
