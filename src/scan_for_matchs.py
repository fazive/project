from os.path import join
from os import walk
import numpy as np
import cv2
from sys import argv
from matplotlib import pyplot as plt
 
# create an array of filenames 
folder = "E:/fazive/studying/py/feature_match_demo/data/roi/roi"
query = cv2.imread(join(folder, "test.jpg"), 0)
 
# create files, images, descriptors globals
files = []
images = []
descriptors = []
for (dirpath, dirnames, filenames) in walk(folder):
	files.extend(filenames)
	for f in files:
		if f.endswith("npy") and f != "test.npy":
			descriptors.append(f)
	print (descriptors)
 
# create the sift detector
sift = cv2.SIFT_create()
query_kp, query_ds = sift.detectAndCompute(query, None)
 
# create FLANN matcher
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
 
# minimum number of matches
MIN_MATCH_COUNT = 10
 
potential_culprits = {}
print (">> 初始化扫描图片...")
for d in descriptors:
	print( "--------- 分析 %s 匹配 ------------" % d)
	matches = flann.knnMatch(query_ds, np.load(join(folder, d)), k =2)
	good = []
	for m,n in matches:
		if m.distance < 0.7*n.distance:
			good.append(m)
	if len(good) > MIN_MATCH_COUNT:
		print ("%s 可匹配! (%d)" % (d, len(good)))
	else:
		print ("%s 不匹配" % d)
	potential_culprits[d] = len(good)
 
max_matches = None
potential_suspect = None
for culprit, matches in potential_culprits.items():
	if max_matches == None or matches > max_matches:
		max_matches = matches
		potential_suspect = culprit
print ("可能匹配的是 %s" % potential_suspect.replace("npy", "").upper())
 
#与测试图片匹配的图片culprit_image
culprit_image = potential_suspect.replace("npy", "jpg")
target = cv2.imread(join(folder, culprit_image), 0)
target_kp, target_ds = sift.detectAndCompute(target, None)
 
#添加文字说明
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(target,'pipeijieguo.jpg ',(10,30), font, 1,(255,0,0),2,cv2.LINE_AA)
cv2.putText(query,'test.jpg',(10,30), font, 1,(255,0,0),2,cv2.LINE_AA)
cv2.imshow('pipeijieguo',target)
 
'''#01暴力匹配BFMatcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(query_ds, target_ds)
matches = sorted(matches, key=lambda x: x.distance)
resultImage = cv2.drawMatches(query, query_kp, target, target_kp, matches[:80], target, flags=2)
'''
#03 获取flann匹配器
matches_o = flann.knnMatch(query_ds, target_ds, k =2)
well= []
for k,l in matches_o:
	if k.distance<0.7*l.distance:
		well.append(k)	
if len(well)>MIN_MATCH_COUNT:
	src_pts = np.float32([ query_kp[k.queryIdx].pt for k in well ]).reshape(-1,1,2)
	dst_pts = np.float32([ target_kp[k.trainIdx].pt for k in well ]).reshape(-1,1,2)
	M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
	matchesMask = mask.ravel().tolist()
	h,w = query.shape
	pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
	dst = cv2.perspectiveTransform(pts,M)
	image = cv2.polylines(target,[np.int32(dst)],True,255,3,cv2.LINE_AA)
else:
	print( "匹配数量不足 - %d/%d" % (len(well),MIN_MATCH_COUNT))
	
	matchesMask = None
 
drawParams=dict(matchColor=(0,255,0),singlePointColor=None,matchesMask=matchesMask,flags=2)
resultImage=cv2.drawMatches(query,query_kp,target,target_kp,well,None,**drawParams)
 
plt.imshow(resultImage)
plt.show()