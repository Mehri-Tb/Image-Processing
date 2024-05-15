import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve, gaussian_filter
from skimage import io

# اين تابع تصویر رنگی RGB را به تصویر خاکستری تبدیل می‌کند.
def R2G(img):
    return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])

#برای کاهش نویز 
def Gauss(img, sigma=1.0):
    return gaussian_filter(img, sigma=sigma)

# تابع برای محاسبه گرادیان با استفاده از فیلتر سوبل
def Filter(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    Gx = convolve(img, Kx)
    Gy = convolve(img, Ky)
    G = np.hypot(Gx, Gy)
    G = G / G.max() * 255
    theta = np.arctan2(Gy, Gx)
    return G, theta

#     این مرحله لبه‌های نازک را حفظ می‌کند و بقیه پیکسل‌های غیر لبه را سرکوب می‌کند.
def NMax(G, theta):
    M, N = G.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = theta * 180.0 / np.pi
    angle[angle < 0] += 180

    for i in range(1, M-1):
        for j in range(1, N-1):
            q = 255
            r = 255
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = G[i, j + 1]
                r = G[i, j - 1]
            elif 22.5 <= angle[i, j] < 67.5:
                q = G[i + 1, j - 1]
                r = G[i - 1, j + 1]
            elif 67.5 <= angle[i, j] < 112.5:
                q = G[i + 1, j]
                r = G[i - 1, j]
            elif 112.5 <= angle[i, j] < 157.5:
                q = G[i - 1, j - 1]
                r = G[i + 1, j + 1]

            if G[i, j] >= q and G[i, j] >= r:
                Z[i, j] = G[i, j]
            else:
                Z[i, j] = 0

    return Z

#     پیکسل‌های قوی، ضعیف و غیر لبه با استفاده از مقادیر آستانه بالا و پایین تشخیص داده می‌شوند.
def Threshold(img, lowThreshold, highThreshold):
    highThreshold = img.max() * highThreshold
    lowThreshold = highThreshold * lowThreshold
    res = np.zeros_like(img)
    strong = np.where(img >= highThreshold)
    weak = np.where((img < highThreshold) & (img >= lowThreshold))
    res[strong] = 255
    res[weak] = 75
    return res

# تابع برای ردیابی لبه‌ها
def Tracking(img):
    M, N = img.shape
    for i in range(1, M-1):
        for j in range(1, N-1):
            if img[i, j] == 75:
                if 255 in img[i-1:i+2, j-1:j+2]:
                    img[i, j] = 255
                else:
                    img[i, j] = 0
    return img

# تابع اصلی برای تشخیص لبه با الگوریتم Canny
def EdgeDetect(img, sigma=1.0, lowThreshold=0.05, highThreshold=0.15):
    gray = R2G(img)
    blurred = Gauss(gray, sigma)
    G, theta = Filter(blurred)
    suppressed = NMax(G, theta)
    Thresholded = Threshold(suppressed, lowThreshold, highThreshold)
    edges = Tracking(Thresholded)
    return edges

# بارگذاری و تست تصویر
img = io.imread('example.jpg')  # تغییر به مسیر صحیح فایل تصویری
edges = EdgeDetect(img)

plt.imshow(edges, cmap='gray')
plt.title('Canny Edge Detection')
plt.show()
