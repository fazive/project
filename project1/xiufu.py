import numpy as np
import cv2
import glob
import os

points = []
def on_mouse(event, x, y, flags, param):
    global img, point1,point2
    img2 = img.copy()
    if event == cv2.EVENT_LBUTTONDOWN:         #左键点击
        point1 = (x,y)
        cv2.circle(img2, point1, 5, (0,0,255),2)
        cv2.imshow('image', img2)
        points.append(list(point1))
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):               #按住左键拖曳
        cv2.rectangle(img2, point1, (x,y), (0,0,255), 2)
        cv2.imshow('image', img2)

    elif event == cv2.EVENT_LBUTTONUP:         #左键释放
        point2 = (x,y)
        cv2.rectangle(img2,point1,point2,(0,0,255), 2)
        cv2.imshow('image', img2)
        points.append(list(point2))
        
if __name__ == '__main__':
    path = glob.glob(os.path.join(r'./static/images/pic', '*.jpg'))
    for file in os.listdir("./static/images/pic"):
        if file.endswith(".jpg"):
            name = file
    for file in path:
        img = cv2.imread(file)
    width = img.shape[0]
    hight = img.shape[1]
    print(img.shape)
    while width >800:
        width = int(width/2)
        hight = int(hight/2)

    img = cv2.resize(img, (hight, width))

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', on_mouse)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    points = np.float64(points)
    inpaintMask = np.zeros(img.shape[:2], np.uint8) #全黑 记录笔画
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inpaintMask[int(points[0][1]):int(points[1][1]),int(points[0][0]):int(points[1][0])] = gray[int(points[0][1]):int(points[1][1]),int(points[0][0]):int(points[1][0])]
    #cv2.imshow('image', inpaintMask)

    res = cv2.inpaint(src=img, inpaintMask=inpaintMask , inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    cv2.imwrite(r'./static/images/regain/'+name, res)
