import cv2
import numpy as np
from panorama_Stitcher import Stitcher

# 预处理函数，去掉合成图像的黑边
def preprocess(path,spath,thre=10000):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   
    column_sum = np.sum(gray, axis=0).reshape(-1,1)
    black_column_indices=0
    print(column_sum.shape)
    for i in range(column_sum.shape[0]):
        #print(column_sum[i][0])
        if column_sum[i][0]>thre:
            black_column_indices+=1
    img2=img[:,0:black_column_indices-1]
    cv2.imwrite(spath, img2)

p1="images/1/yosemite2.jpg"
p2="images/1/yosemite1.jpg"
p3="images/1/yosemite4.jpg"
p4="images/1/yosemite3.jpg"

res_path1="results/1/n1.png"
res_path2="results/1/n2.png"
res="results/1/n3.png"


stitcher1 = Stitcher(p1,p2,res_path1)
stitcher1.stitch()
preprocess(res_path1,res_path1)

stitcher2 = Stitcher(p3,p4,res_path2)
stitcher2.stitch()
preprocess(res_path2,res_path2)


stitcher3 = Stitcher(res_path2,res_path1,res)
stitcher3.stitch()
preprocess(res,res)
#stitcher.match()
#stitcher.HOG_kps_match()


