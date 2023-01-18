import numpy as np
import cv2


def cv2imread(img_path):    #读取图片
    return cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)


def cv2imwrite(filename, img):  # 保存图片
    cv2.imencode('.png', img)[1].tofile(filename)
