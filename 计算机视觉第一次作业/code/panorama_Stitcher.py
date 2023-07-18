import numpy as np
import cv2
from tqdm import trange

class Stitcher:
    def __init__(self,path1,path2,res):
        self.img1=cv2.imread(path1)
        self.img2 =cv2.imread(path2)
        self.kp1=None
        self.kp2=None
        self.resp=res

    def SIFI_kps_match(self,path):
        print("using SIFI to match keypoints...")
        # 创建SIFT对象
        sift = cv2.SIFT_create()
        gray1 = cv2.cvtColor(self.img2, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
        # 检测关键点和计算描述符
        kp1, des1 = sift.detectAndCompute(gray1,None)
        kp2, des2 = sift.detectAndCompute(gray2,None)
        # 创建BFMatcher对象
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        # 匹配关键点描述符
        matches = bf.match(des1,des2)
        # 根据特征点匹配程度排序
        matches = sorted(matches, key = lambda x:x.distance)
        # 绘制匹配结果
        #print(matches)
        result = cv2.drawMatches(self.img2,kp1,self.img1,kp2,matches[:],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # 保存匹配结果图像
        cv2.imwrite(path,result)
        self.kp1=kp1
        self.kp2=kp2

    def stitch(self, ratio=0.75, Thresh=4.0):
        
        # 检测A B特征关键点，并计算特征描述子
        kps1, des1 = self.detect_compute_plus(self.img1)
        kps2, des2 = self.detect_compute_plus(self.img2)
        # 匹配两张图片的所有特征点，返回匹配结果
        matches, H, status = self.match_kps(kps1, kps2, des1, des2, ratio, Thresh)
        result = cv2.warpPerspective(self.img1, H, (self.img1.shape[1] + self.img2.shape[1], self.img1.shape[0]))
        result[0:self.img2.shape[0], 0:self.img2.shape[1]] = self.img2
        cv2.imwrite(self.resp,result)
        return result

    def detect_compute_plus(self, img):
        # 将图像转为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 创建SIFT检测器
        descripter = cv2.SIFT_create()
        # 在图像上检测关键点并提取特征
        (kps, features) = descripter.detectAndCompute(img, None)
        # 将关键点的坐标转为numpy数组
        kps = np.float32([kp.pt for kp in kps])
        return kps, features

    def match_kps(self, kps1, kps2, des1, des2, ratio, Thresh):
        # 创建BFMatcher对象
        matcher = cv2.BFMatcher()
        # 对两张图像的特征进行匹配，得到匹配结果列表
        match_lists = matcher.knnMatch(des1, des2, 2)
        # 进行筛选
        matches = []
        for m in match_lists:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))
        # 根据匹配结果，计算变换矩阵，并返回匹配点列表、变换矩阵、状态
        if len(matches) > 4:
            ptsA = np.float32([kps1[i] for (_, i) in matches])
            ptsB = np.float32([kps2[i] for (i, _) in matches])
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, Thresh)
            return matches, H, status


    def HOG_kps_match(self,path):
        print("using HOG to match keypoints...")
        # 获取输入的特征点
        kp1=self.kp1
        kp2=self.kp2
        corners1 = []
        corners2 = []
        for kp in kp1:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            corners1.append((x,y))
        for kp in kp2:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            corners2.append((x,y))
        # HOG算法参数 
        winSize = (64, 64)
        blockSize = (16, 16)
        blockStride = (8, 8)
        cellSize = (8, 8)
        nbins = 9
        # 初始化HOGDescriptor对象
        hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
        # 获取第一幅图像的HOG特征描述符
        descriptors1 = []
        cnt=0
        for corner in corners1:
            (x, y) = corner
            if(y-32>=0 and y+32<self.img1.shape[0] and x-32>=0 and x+32<self.img1.shape[1]):
                cnt+=1
                patch = self.img1[y-32:y+32, x-32:x+32]
                descriptors1.append(hog.compute(patch))
        # 获取第二幅图像的HOG特征描述符
        descriptors2 = []
        cnt=0
        for corner in corners2:
            (x, y) = corner
            if(y-32>=0 and y+32<self.img2.shape[0] and x-32>=0 and x+32<self.img2.shape[1]):
                cnt+=1
                patch = self.img2[y-32:y+32, x-32:x+32]
                descriptors2.append(hog.compute(patch))
        matches = []
        tag=[]
        for i in trange(len(descriptors1)):
            descriptor1=descriptors1[i]
            best_match = -1
            best_distance = float('inf')
            for j, descriptor2 in enumerate(descriptors2):
                distance = np.linalg.norm(descriptor1 - descriptor2,2)
                if distance < best_distance and (j not in tag):
                    best_match = j
                    tag.append(j)
                    best_distance = distance
            if best_match!=-1:
                matches.append((i, best_match, best_distance))
        # 按照距离从小到大排序
        matches = sorted(matches, key=lambda x: x[2])
        b=[(m[0],m[1]) for m in matches]
        print(b)
        a=[cv2.DMatch(m[0], m[1], m[2]) for m in matches]
        img_matches = cv2.drawMatches(self.img1, kp1, self.img2, kp2, a[:], None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite(path, img_matches)


        

