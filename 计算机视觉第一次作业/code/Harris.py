import cv2
import numpy as np

SIZE=3
sigma=1
k=0.04
threshold=1e-6
maxnum=1000
def detect_corners(img):
    # 计算梯度
    dx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=SIZE)
    dy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=SIZE)
    # 计算结构张量
    dx2 = dx * dx
    dy2 = dy * dy
    dxy = dx * dy
    # 高斯加权平均
    weights = cv2.getGaussianKernel(SIZE, sigma)
    Sx2 = cv2.filter2D(dx2, -1, weights)
    Sy2 = cv2.filter2D(dy2, -1, weights)
    Sxy = cv2.filter2D(dxy, -1, weights)
    # 计算 Harris 值
    detM = Sx2 * Sy2 - Sxy * Sxy
    traceM = Sx2 + Sy2
    Harris = k * detM / (traceM + 1e-12)
    # 选取 Harris 值最大的像素点作为角点
    corner_list = []
    offset = SIZE // 2
    for i in range(offset, img.shape[0] - offset):
        for j in range(offset, img.shape[1] - offset):
            if Harris[i, j] > threshold and Harris[i, j] == np.max(Harris[i-offset:i+offset+1, j-offset:j+offset+1]):
                corner_list.append((j, i))   
    # 如果角点数量超过最大值，z则选前 maxnum 个角点
    if len(corner_list) > maxnum:
        corner_list = sorted(corner_list, key=lambda x: Harris[x[1], x[0]], reverse=True)[:maxnum]
    return corner_list


idx=0
png_list=[
    'images/1/sudoku.png',
    'images/1/uttower1.jpg',
    'images/1/uttower2.jpg'
]
save_list=[
    'results/1/sudoku_keypoints.png',
    'results/1/uttower1_keypoints.jpg',
    'results/1/uttower2_keypoints.jpg'
]

img = cv2.imread(png_list[idx])

# 转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# 对图像进行角点检测
corner_list = detect_corners(gray)

# 绘制角点
for corner in corner_list:
    cv2.circle(img, corner, 3, (0, 0, 255), 2)
    
 
# 保存角点检测结果
cv2.imwrite(save_list[idx], img)
