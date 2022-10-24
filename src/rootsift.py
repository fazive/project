import json
import os
import pickle
import traceback
from datetime import datetime
from os import walk
from os.path import join

import cv2
import numpy as np


class RootSIFT:
    ''' def __init__(self) -> None: 增加代码可读性，表示返回的是None，但是实际return也ok
        创建一个SIFT_create()的提取机
    '''
    def __init__(self) -> None:
        self.extractor = cv2.SIFT_create()
        
    '''
    函数找到图像中的关键点。
    如果您只想搜索图像的一部分，则可以传递掩码。
    每个关键点都是一个特殊的结构，它具有许多属性，
    例如它的 (x,y) 坐标、有意义的邻域的大小、指定其方向的角度、指定关键点强度的响应等。
    '''
    def detect(self, img):
        return self.extractor.detect(img)
    '''
    由于您已经找到了关键点，您可以调用sift.compute()从我们找到的关键点计算描述符。
    例如：kp,des = sift.compute(gray,kp)
    检测和计算
    '''
    def compute(self, img, eps=1e-7):
        kps = self.detect(img)
        if len(kps) == 0:
            return([], None)
        #  kp 将是关键点列表，而 des 是形状的 numpy 数组（关键点数）× 128
        (kps, descs) = self.extractor.compute(img, kps)
        # if there are no keypoints or descriptors, return an empty tuple
        if len(kps) == 0:
            return([], None)
        # apply the Hellinger kernel by first L1-normalizing and taking the
        # square-root 
        # 归一化
        descs /= (descs.sum(axis=1, keepdims=True) + eps)
        descs = np.sqrt(descs)
        #descs /= (np.linalg.norm(descs, axis=1, ord=2) + eps)
        # return a tuple of the keypoints and descriptors
        return (kps, descs)

def read_confs(confs_path):
    '''
    该方法用于读取配置文档
    '''
    f = open(confs_path, 'r')
    settings = json.load(f)
    data = f'{str(confs_path)}已读取\n'
    print(data)
    write_logs(data)
    return settings['ratio_thresh'], settings['save_model_folder'], settings['template_path'], settings['src_path'], settings['crop_images_path'],settings['roi']

def get_image_list(src_path):
    '''
    获取当前文件夹的图片列表
    '''
    files = []
    images = []
    write_logs(f'{src_path}正在获取文件列表')
    for (dirpath, dirnames, filenames) in walk(src_path):
        files.extend(filenames)
        for f in files:
            if f.endswith("jpg"):
                images.append(f'{src_path}{f}')
                write_logs(f'{src_path}{f}')
        # print (images)
        
    return images

def crop_image(img, crop_images_path, roi):
    '''
    分割图片中roi区域，并储存
    '''
    x1 = roi['roi'][0]['x1']
    x2 = roi['roi'][0]['x2']
    y1 = roi['roi'][0]['y1']
    y2 = roi['roi'][0]['y2']
   
    img_n = img[y1:y2, x1:x2, :]

    now = datetime.now()

    folder_path_now = now.strftime('%Y%m%d%H%M%S')
    # print(time_now)
    name = roi['name']
    image_path = f'{crop_images_path}{name}{folder_path_now}.jpg'
    print(image_path)
    cv2.imwrite(image_path, img_n)

    write_logs(f'图片{image_path}已切割\n')

    return image_path

def get_model_path(save_model_folder, image_path):
    '''
    获取该图片的模型路径
    '''
    model_path = f"{save_model_folder}/{image_path.split('/')[-1]}".replace("jpg", "pkl")
    write_logs(f'已获取对应模型路径{model_path}\n')
    return model_path

def check_path(model_path):
    '''
    检测图片路径是否存在
    '''
    return os.path.exists(model_path)

def save_model(model_path, keypoints, descriptors):
    '''
    保存模型：
    '''
    keypoints_dict = []
    for i in range(len(keypoints)):
        keypoints_message = {'pt':keypoints[i].pt , 'size':keypoints[i].size , 'angle':keypoints[i].angle ,'octave':keypoints[i].octave ,
                            'class_id':keypoints[i].class_id}#把这些信息写入到字典中
        keypoints_dict.append(keypoints_message)
    pickle.dump([keypoints_dict, descriptors], open(model_path,'wb'))
    write_logs(f'{model_path}模型成功保存\n')

def restore_keypoints_format():
    '''
    将字典类型的keypoints还原
    '''

    return 0

def create_model(feature_detector, image_path, model_path):
    '''
    创建模型
    '''
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (keypoints, descriptors) = feature_detector.compute(gray_image)
    print(f'{model_path}模型已创建\n')
    write_logs(f'{model_path}模型已创建\n')
    save_model(model_path, keypoints, descriptors)
    return (keypoints, descriptors)

def load_model(feature_detector, image_path, model_path):
    '''
    加载模型：先判断模型是否存在，存在->加载，不存在->创建并返回信息
    '''
    if check_path(model_path):
        print(f'{model_path}模型正在加载')        
        model = pickle.load(open(model_path,'rb'))
        # data = f'——————————{model_path}已加载——————————\n'
        # print(np.load(model_path))
        write_logs(f'{model_path}已加载\n')
        return model
    else:
        # print(f'——————————{model_path}不存在，正在创建——————————\n')
        # data = f'——————————{model_path}模型不存在，正在创建——————————\n'
        write_logs(f'{model_path}模型不存在，正在创建\n')
        keypoints, descriptors = create_model(feature_detector, image_path, model_path)
        return [keypoints, descriptors]




'''保存文件'''        
""" def save_descriptor(image_path, descriptors, save_model_folder):
    print( f"读取中... {image_path}")    
    descriptor_file = f"{save_model_folder}/{image_path.split('/')[-1]}".replace("jpg", "npy")                                                            
    # descriptor_file = image_path.replace("jpg", "npy")
    # image_path.split('/')[-1]
    np.save( descriptor_file,  descriptors)
 """

def draw_key_points(img, kps, wait_sec = 3):
    '''
    OpenCV 还提供了cv.drawKeyPoints()函数，它在关键点的位置上绘制小圆圈。
    如果您将标志cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS传递给它，它将绘制一个关键点大小的圆，甚至会显示其方向
    '''
    # 如果kps并不是list类型则返回，如果img不是np.ndarray则返回
    write_logs('——————————draw_key_points——————————\n')
    if not isinstance(kps, list):
        write_logs('——————————kps format should be a list.——————————')
        return {'res': None, 'info': 'kps format should be a list.'}

    if not isinstance(img, np.ndarray):
        write_logs('——————————image format should be a numpy array.——————————')
        return {'res': None, 'info': 'image format should be a numpy array.'}

    original = img.copy()
    # 数组新的shape属性应该要与原来的配套，
    # 如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。
    if img.shape[-1] != 1:
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except:
            write_logs(f'{traceback.format_exc()}\n')
            return {'res': None, 'info': f"{traceback.format_exc()}"}

    if len(kps) == 0:
        write_logs('——————————Can not find key points.——————————')
        return {'res': None, 'info': 'Can not find key points.'}
    
    cv2.drawKeypoints(img, kps, original, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.namedWindow('kp', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('kp', original)
    cv2.waitKey(wait_sec)
    write_logs(f'——————————展示图片结束——————————\n')

def write_logs(data, logs_folder = 'E:/fazive/studying/py/feature_match_demo/logs'):
    '''
    该方法用于编写log日志
    '''
    now = datetime.now()
    time_now = now.strftime('%m/%d/%Y, %H:%M:%S')
    folder_path_now = now.strftime('%Y%m%d')
    # print(time_now)
    logs_folder_path = f'{logs_folder}/logs_{folder_path_now}.txt'
    with open(logs_folder_path, 'a+', encoding='utf-8') as f:
        f.write(f'{time_now} :\n {data}\n')


