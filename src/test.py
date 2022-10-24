from datetime import datetime

import cv2

from rootsift import read_confs

print("请输入project_id('AC1.5'OR'DL380'):\n")
project_id = input()
# print(project_id)
confs_path = f"E:/fazive/studying/py/feature_match_demo/data/{project_id}/result/{project_id}.json"

ratio_thresh, save_model_folder, template_path, src_path, roi = read_confs(confs_path)
print(roi['name'], roi['roi'][0]['x1'])

x1 = roi['roi'][0]['x1']
x2 = roi['roi'][0]['x2']
y1 = roi['roi'][0]['y1']
y2 = roi['roi'][0]['y2']
img = cv2.imread('./data/AC1.5/result/output/2022-10-23/NoRead_1799_top1.jpg')
img_n = img[y1:y2, x1:x2, :]

now = datetime.now()
time_now = now.strftime('%m/%d/%Y, %H:%M:%S')
folder_path_now = now.strftime('%Y%m%d%H%M%S')
# print(time_now)
name = roi['name']
image_path = f'./data/AC1.5/result/output/2022-10-23/{name}{folder_path_now}.jpg'
print(image_path)
cv2.imwrite(image_path, img_n)
