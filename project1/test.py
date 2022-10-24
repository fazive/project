# -*- coding: utf-8 -*-
"""
Created on Thu May 20 10:57:05 2021

@author: Lenovo
"""

import paddlex as pdx
import matplotlib.pyplot as plt
from paddlex.det import transforms
import os
from werkzeug.utils import secure_filename
pls = ['pl40','pl5','pl10','pl15','pl20','pl25','pl30','pl35','pl50','pl60','pl70','pl80','pl90','pl100','pl110','pl120']
Is = ['i5','i6','i7','i8','i10','i11','i12','i13','i14','i15']
ps = ['p5','p14','p19','p20','p21','p23','p28']
pas = ['pa8','pa10','pa12','pa13','pa14']
ph = ['ph1.5','ph3.5','ph2.1','ph2.2','ph2.4','ph2.5','ph2.8',
     'ph2.9','ph2','ph3.2','ph3','ph4.2','ph4.3','ph4.5','ph4.8'
     'ph4','ph5.3','ph5']  #287
pm = ['pm10','pm2','pm5','pm8','pm13','pm15','pm20','pm30','pm35',
     'pm40','pm50','pm55'] #52
def select(test_jpg,result):
    for i in result:
        if  i['category'] in pls:
            i['category'] = 'pl'
        elif i['category'] in Is:
            i['category'] = 'it'
        elif i['category'] in ps:
            i['category'] = 'pt'
        elif i['category'] in pas:
            i['category'] = 'pa'
        elif i['category'] in ph:
            i['category'] = 'ph'
        elif i['category'] in pm:
            i['category'] = 'pm'
    from pygame import mixer 
    import time
    sound = []
    
    for i in result:
        if i['score']>0.02:
            sound.append(i['category'])
    for i in sound:
        mixer.init()
        mixer.music.load('sound/'+ i + '.mp3')
        mixer.music.play()
        time.sleep(1.3)
        mixer.music.stop()
#      print(result)
#      os.system('python demo_two_stream.py')
#              end = tuple(i for i in results)
#              print(end)
        pdx.det.visualize(test_jpg, result, threshold=0.02, save_dir='./pic')

#pdx.det.visualize(test_jpg, result, threshold=0.05, save_dir='./output/image')
eval_transforms = transforms.Compose([
    transforms.Resize(target_size=608, interp='CUBIC'), 
    transforms.Normalize()
])
#model = pdx.load_model('./16epoch/')
#test_jpg = '2797.jpg'
#result = model.predict(test_jpg,eval_transforms)
model = pdx.load_model('./41epoch')
for file in os.listdir(r".\new/"):
        if file.endswith(".jpg"):
            file_name = secure_filename(file)
            test_jpg  = r'.\new/'+file
            print(test_jpg)
            result = model.predict(test_jpg,eval_transforms)
#            print(result)
            select(test_jpg,result)
            
           
#print(len(result))
#sound = []
#for i in result:
#    if i['score']>0.1:
#        sound.append(i['category'])
#from pygame import mixer 
#import time
#for i in sound:
#    mixer.init()
#    mixer.music.load('sound/'+ i + '.mp3')
#    mixer.music.play()
#    time.sleep(4)
#    mixer.music.stop()
#print(sound)
#    break