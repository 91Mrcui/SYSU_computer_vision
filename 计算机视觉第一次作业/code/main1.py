import cv2
from panorama_Stitcher import Stitcher

# 要拼接的图像
p1="images/1/uttower2.jpg"
p2="images/1/uttower1.jpg"

# 关键点匹配结果保存路径
match_path="uttower_match.png"
# 拼接结果保存路径
res="uttower_stitching_sift.png"

stitcher = Stitcher(p1,p2,res)
# HOG作为特征描述子
stitcher.SIFI_kps_match(match_path)

# HOG作为特征描述子
stitcher.HOG_kps_match(match_path)

# 拼接
stitcher.stitch()




