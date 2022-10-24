# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 14:32:42 2020

@author: Lenovo
"""
from flask import Flask, render_template, request, redirect, Response
from werkzeug.utils import secure_filename
import os
import glob
import base64
import numpy as np
import paddlex as pdx
from paddlex.det import transforms
import cv2
import matplotlib.pyplot as plt
app = Flask(__name__,static_folder='static')
# 设置图片保存文件夹
# UPLOAD_FOLDER = 'photo'
UPLOAD_FOLDER = r'.\static\images\pic'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
eval_transforms = transforms.Compose([
    transforms.Resize(target_size=608, interp='CUBIC'), 
    transforms.Normalize()
])
model = pdx.load_model('./41epoch/')
# 设置允许上传的文件格式
ALLOW_EXTENSIONS = ['png', 'jpg', 'mp4']

# 判断文件后缀是否在列表中
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[-1] in ALLOW_EXTENSIONS

@app.route('/')
def index():
    return  render_template('水印管家 一键清除视频&图片水印.html')
@app.route('/e/')
def unable():
    path_list = []
    path_list.append(r'D:\notebook\项目实训\项目实训（最终版\static\images\pic')
    path_list.append(r'D:\notebook\项目实训\项目实训（最终版\static\images\mask')
    path_list.append(r'D:\notebook\项目实训\项目实训（最终版\static\images\box')
    path_list.append(r'D:\notebook\项目实训\项目实训（最终版\static\images\mp4')
    path_list.append(r'D:\notebook\项目实训\项目实训（最终版\static\images\regain')
    paths = []
    for path in path_list:
        for i in ALLOW_EXTENSIONS:
            paths.append(glob.glob(os.path.join(path, '*.' + i)))
    for files in paths:
        for file in files:
            if allowed_file(file):
                os.remove(file)
    return render_template('检测.html')
@app.route('/login/')
def login():
    # 一般情况， 不会直接把html文件内容直接返回；
    # 而是将html文件保存到当前的templates目录中；
    #       1). 通过render_template方法调用;
    #       2). 默认情况下,Flask 在程序文件夹中的 templates 子文件夹中寻找模板。
    return  render_template('login.html')

@app.route('/login2/')
def login2():
    # 获取用户输入的用户名
    username = request.args.get('username')
    password = request.args.get('password')
    print(username,password)
    # 逻辑处理， 用来判断用户和密码是否正确;
    if username == 'root' and password == 'redhat':
        # 重定向到指定路由；
        return  redirect('/')
        # return "登录成功"
    else:
        return  "登录失败"

def return_img_stream(img_local_path):
  """
  工具函数:
  获取本地图片流
  :param img_local_path:文件单张图片的本地绝对路径
  :return: 图片流
  """
  with open(img_local_path, 'rb') as img_f:
    print(img_f)
    img_stream = img_f.read()
    img_stream = base64.b64encode(img_stream)
  return img_stream

@app.route('/register/')
def register():
    return render_template('register.html')

# 上传图片
@app.route("/u/", methods=['POST', "GET"])
def uploads():
  if request.method == 'POST':
    # 获取post过来的文件名称，从name=file参数中获取
    file = request.files['file']
    if file and allowed_file(file.filename):
      
      # secure_filename方法会去掉文件名中的中文
      file_name = secure_filename(file.filename)
      # 保存图片
#      print(filename)
      file.save(os.path.join(app.config['UPLOAD_FOLDER'], file_name))
      result = model.predict( r'.\static\images\pic/'+file_name,eval_transforms)
#      print(result)

      
      pls = ['pl40','pl5','pl10','pl15','pl20','pl25','pl30','pl35','pl50','pl60','pl70','pl80','pl90','pl100','pl110','pl120']
      Is = ['i5','i6','i7','i8','i10','i11','i12','i13','i14','i15']
      ps = ['p5','p14','p19','p20','p21','p23','p28']
      pas = ['pa8','pa10','pa12','pa13','pa14']
      ph = ['ph1.5','ph3.5','ph2.1','ph2.2','ph2.4','ph2.5','ph2.8',
           'ph2.9','ph2','ph3.2','ph3','ph4.2','ph4.3','ph4.5','ph4.8'
           'ph4','ph5.3','ph5']  #287
      pm = ['pm10','pm2','pm5','pm8','pm13','pm15','pm20','pm30','pm35',
           'pm40','pm50','pm55'] #524

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
          if i['score']>0.5:
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
      pdx.det.visualize(r'.\static\images\pic/'+file_name, result, threshold=0.3, save_dir=r'.\static\images\box')
      img1 = glob.glob(os.path.join(r'.\static\images\pic', '*.jpg'))
      img2 = glob.glob(os.path.join(r'.\static\images\box', '*.jpg'))
      print(result)
      return render_template("检测.html", img1 = img1, img2 = img2)
    else:
      return "格式错误，请上传jpg或png格式文件"


  return render_template('检测.html')

@app.route("/video/", methods=['POST', "GET"])
def video():
  if request.method == 'POST':
    # 获取post过来的文件名称，从name=file参数中获取
    file = request.files['file']
    if file and allowed_file(file.filename):
      print(file.filename)
      # secure_filename方法会去掉文件名中的中文
      file_name = secure_filename(file.filename)
      # 保存图片
      file.save(os.path.join('.\static\images\mp4', file_name))
      os.system('python shipin.py')
      os.system('python demo_two_stream.py')
      img = glob.glob(os.path.join(r'.\static\images\box', '*.jpg'))
      return render_template("video.html", img=img)
    else:
      return "格式错误，请上传mp4格式文件"
  return render_template('video.html')

@app.route("/regain/", methods=['POST', "GET"])
def regain():
  if request.method == 'POST':
    # 获取post过来的文件名称，从name=file参数中获取
    file = request.files['file']
    if file and allowed_file(file.filename):
      print(file.filename)
      # secure_filename方法会去掉文件名中的中文
      file_name = secure_filename(file.filename)
      # 保存图片
      file.save(os.path.join(app.config['UPLOAD_FOLDER'], file_name))
      os.system('python xiufu.py')
      img1 = glob.glob(os.path.join(r'.\static\images\pic', '*.jpg'))
      img2 = glob.glob(os.path.join(r'.\static\images\regain', '*.jpg'))
      return render_template("检测.html", img1=img1, img2=img2)
    else:
      return "格式错误，请上传jpg或png格式文件"
  return render_template('regain.html')


# 查看图片
@app.route("/upload/<imageId>.jpg")
def get_frame(imageId):
    # 图片上传保存的路径
    print(imageId)
    with open(r'D:\notebook\项目实训\项目实训（最终版\static\images\box\{}.jpg'.format(imageId), 'rb') as f:
        image = f.read()
        resp = Response(image, mimetype="image/jpg")
        return resp

if __name__ == '__main__':
    app.run('localhost', 4000,debug=True)
