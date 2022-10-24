import os
import sys

import cv2

from rootsift import RootSIFT, create_model, get_model_path, read_confs, write_logs, get_image_list, crop_image

# settings
write_logs('-------------------STARTING----------------------\n')
rs = RootSIFT()
print("请输入project_id('AC1.5'OR'DL380'):\n")
project_id = input()
# print(project_id)
confs_path = f"E:/fazive/studying/py/feature_match_demo/data/{project_id}/result/{project_id}.json"
write_logs(f'——————————{confs_path}已加载——————————')
# print(confs_path)


ratio_thresh, save_model_folder, template_path, src_path, crop_images_path, roi = read_confs(confs_path)


src_image_list = get_image_list(src_path)
template_image_list = get_image_list(template_path)

for src_path in src_image_list:  
    for  template_path in template_image_list:
        '''
        待完成：判断模型是否存在，存在则加载模型，不存在创建模型
        '''
        # 判断模型是否存在
        
        # load template
        # print(os.path.dirname(os.path.abspath(template_path)))
        template = cv2.imread(template_path)
        gray_tmpl = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        # load src image
        src = cv2.imread(src_path)
            
        src_crop_path = crop_image(src, crop_images_path, roi)
        src_crop = cv2.imread(src_crop_path)
        gray_src = cv2.cvtColor(src_crop, cv2.COLOR_BGR2GRAY)

        # detect feature points
        # kp 将是关键点列表，而 des 是形状的 numpy 数组（关键点数）× 128

        template_model_path = get_model_path(save_model_folder, template_path)
        (kps_template, features_template) = create_model(rs, template_path, template_model_path)

        # (kps_template, features_template) = rs.compute(gray_tmpl)
        (kps_src, features_src) = rs.compute(gray_src)
        # src_model_path = get_model_path(save_model_folder, src_crop_path)

        # (kps_src, features_src) = create_model(rs, src_crop_path, src_model_path)
        # match features


        matcher = cv2.DescriptorMatcher_create("FlannBased")
        knn_matches = matcher.knnMatch(features_template, features_src, 2)
        # print(knn_matches)

        good = []
        for m,n in knn_matches:
            # print(m,n)
            if m.distance < ratio_thresh * n.distance:
                good.append([m])
                # print(good)
            # break
        print(good)

        display = cv2.drawMatchesKnn(template, kps_template, 
                                    src_crop, kps_src,
                                    good,  None, flags=2)

        # display result image
        cv2.namedWindow('display', cv2.WINDOW_NORMAL)
        cv2.imshow('display', display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if good: 
            break
write_logs('-------------------ENDING----------------------\n')
