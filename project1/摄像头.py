# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 20:44:52 2020
@author: 指尖魔法师
功能：python调用手机摄像头，并保存视频
说明：
手机需要安装IP摄像头app
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python
"""
import cv2
import time
 
video ='http://admin:admin@192.168.111.189:8081'
capture = cv2.VideoCapture(video)
 
if capture.isOpened():
    cv2.namedWindow('camera',cv2.WINDOW_NORMAL)
    
#    保存avi视频    q
#    fourcc = cv2.VideoWriter_fourcc(*'XVID')
#    # fps = cap.get(cv2.CAP_PROP_FPS)
#    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
#    out = cv2.VideoWriter('camera_test.avi', fourcc,10.0, size)
 
#    保存MP4视频     
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    fps = 30
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter()
    out.open('output.mp4',fourcc,fps,size,True)
 
    while capture.isOpened():
        ret,frame = capture.read()
        if ret:
#            #设定了灰值后无法保存视频
#            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
            cv2.imshow('camera',frame)
            
            out.write(frame)
            
            #按空格保存图像，按esc退出
            key = cv2.waitKey(10) 
            if key == 27:
                break
            
            if key == ord(' '):
                photoname=str(int(time.time()))+'.jpg'
                cv2.imwrite(photoname,frame)
        else:
            break;
    
    
    
    capture.release()
    out.release()
    cv2.destroyAllWindows()