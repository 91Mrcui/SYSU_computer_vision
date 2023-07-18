import sys
import cv2
import numpy as np
from scipy import ndimage

DEPTH = 3  # 图像通道数

#构建索引数组
def make_seam_index(energy_map, height, zeroArr):
    binaryArr = np.ones_like(energy_map, dtype=bool)
    all_seams = []
    b = np.argmin(energy_map[-1])
    for a in range(-1, height - 1):
        binaryArr[a, b] = False
        all_seams.append(b)
        b = zeroArr[a, b]
    all_seams.reverse()
    all_seams = np.array(all_seams)
    return all_seams, binaryArr

# 寻找最小能量的缝
def find_min_seam(im):
    height, width = im.shape[:2]
    grad_x = ndimage.convolve1d(im, np.array([1, 0, -1]), axis=1, mode='wrap')  # 计算图像水平方向上的梯度
    grad_y = ndimage.convolve1d(im, np.array([1, 0, -1]), axis=0, mode='wrap')  # 计算图像垂直方向上的梯度
    energy_map = np.sqrt(np.sum(grad_x ** 2, axis=2) + np.sum(grad_y ** 2, axis=2))  # 计算能量图
    h, w = 0, 1
    zeroArr = np.zeros((height, width), dtype=int)
    for a in range(w, height):
        for b in range(h, width):
            if not b == h:
                Ix = [np.argmin(energy_map[a - w, b - w:b + 2])]  # 获取最小能量路径
                zeroArr[a, b] = Ix[0] + b - w
                temp_energy = energy_map[a - w, Ix[0] + b - w]
            else:
                Ix = [np.argmin(energy_map[a - w, b:b + 2])]  # 获取最小能量路径
                zeroArr[a, b] = Ix[0] + b
                temp_energy = energy_map[a - w, Ix[0] + b]
            energy_map[a, b] += temp_energy
    return make_seam_index(energy_map, height, zeroArr)

# 创建缝列表
def create_seams_list(img, n):
    seams_vec = []
    current_img = np.array(img)
    i = 0
    while i < n:
        Ix, binary_im = find_min_seam(current_img)
        seams_vec.extend([Ix])  # 添加最小能量路径
        current_img = current_img[np.stack([binary_im] * DEPTH, axis=2)].reshape(
            (current_img.shape[:2][0], current_img.shape[:2][1] - 1, DEPTH))  # 移除最小能量路径对应的像素
        i += 1
    seams_vec.reverse()
    return seams_vec

def get_prepended_pixels(img, a, b, c):
    return img[a, :b, c]  # 获取指定通道的图像左侧像素值

def get_appended_pixels(img, a, b, c):
    return img[a, b:, c]  # 获取指定通道的图像右侧像素值

# 将缝插入图像中
def place_seam(currentSeam, img):
    height, width, DEPTH = img.shape
    arry = np.zeros((height, 1 + width, DEPTH))
    for vertical in range(height):
        horizontal = currentSeam[vertical]
        for color in range(DEPTH):
            if not horizontal == 0:
                avaeragePixel = np.average(img[vertical, horizontal - 1: horizontal + 1, color])
                arry[vertical, horizontal, color] = avaeragePixel

                arry[vertical, : horizontal, color] = get_prepended_pixels(img, vertical, horizontal, color)  # 在路径左侧放置像素值
                arry[vertical, horizontal + 1:, color] = get_appended_pixels(img, vertical, horizontal, color)  # 在路径右侧放置像素值
            else:
                avaeragePixel = np.average(img[vertical, horizontal: horizontal + 2, color])
                arry[vertical, horizontal + 1, color] = avaeragePixel

                arry[vertical, horizontal, color] = img[vertical, horizontal, color]
                arry[vertical, horizontal + 1:, color] = get_appended_pixels(img, vertical, horizontal, color)  # 在路径右侧放置像素值
    return arry

# 添加seam
def add_seams(n, img):
    seams_vec = create_seams_list(img, n)
    i = 0
    while i < n:
        seam = seams_vec[-1]
        seams_vec = seams_vec[:-1]
        img = place_seam(seam, img)
        for current in seams_vec:
            current[np.where(current >= seam)] += 2
        i += 1
    return img

# 增大图像
def enlarge_img(horizontal_seams, vertical_seams, img):
    res = img
    if horizontal_seams > 0:
        res = add_seams(horizontal_seams, res)  # 放置水平方向的路径
    if vertical_seams > 0:
        res = np.rot90(res, 1)  # 顺时针旋转90度
        res = add_seams(vertical_seams, res)  # 放置垂直方向的路径
        res = np.rot90(res, 3)  # 逆时针旋转90度
    return res

# 消除缝
def remove_seams(n, img):
    cnt = 0
    n = - n
    while cnt < n:
        mins = find_min_seam(img)
        height, width = img.shape[:2]
        mask = np.stack([mins[1]] + [mins[1]] + [mins[1]], axis=2)  # 生成布尔掩码
        img = img[mask].reshape((height, width - 1, DEPTH))  # 移除最小能量路径对应的像素
        cnt += 1
    return img

# 缩小图像
def shrinking_img(horizontal_seams, vertical_seams, img):
    res = img
    if horizontal_seams < 0:
        res = remove_seams(horizontal_seams, res)  # 移除水平方向的路径
    if vertical_seams < 0:
        res = np.rot90(res, 1)  # 顺时针旋转90度
        res = remove_seams(vertical_seams, res)  # 移除垂直方向的路径
        res = np.rot90(res, 3)  # 逆时针旋转90度
    return res

if __name__ == '__main__':
    if not len(sys.argv) == 6:
        print("Wrong input")
        exit()
    img = cv2.imread(sys.argv[1])  # 读取输入图像
    horizontal_seams = int(sys.argv[4])  # 获取水平路径数量
    vertical_seams = int(sys.argv[5])  # 获取垂直路径数量
    if sys.argv[3] == "enlarge":
        output = enlarge_img(horizontal_seams, vertical_seams, img)  # 执行图像放大操作
    elif sys.argv[3] == "shrinking":
        output = shrinking_img(-horizontal_seams, -vertical_seams, img)  # 执行图像缩小操作
    else:
        print("invalid resize type")
        exit()
    cv2.imwrite(sys.argv[2], output)  # 保存输出图像
