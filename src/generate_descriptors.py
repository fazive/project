import cv2
import numpy as np
from os import walk
from os.path import join

 
def create_descriptors(folder):
	files = []
	for (dirpath, dirnames, filenames) in walk(folder):
		files.extend(filenames)
	for f in files:
		save_descriptor(folder, f, cv2.SIFT_create())
        # save_descriptor(folder, f, cv2.xfeatures2d.SIFT_create())
		
def save_descriptor(folder, image_path, feature_detector):
	print( "读取中... %s" % image_path)
	# if image_path.endswith("npy"):
	# 	return
	img = cv2.imread(join(folder, image_path), 0)
	keypoints, descriptors = feature_detector.detectAndCompute(img, None)
	descriptor_file = image_path.replace("jpg", "npy")
	np.save(join(folder, descriptor_file), descriptors)
	
create_descriptors("E:/fazive/studying/py/feature_match_demo/data/roi/roi")