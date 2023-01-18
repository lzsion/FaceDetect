import cv2
import sys
import os

from FaceDetect.settings import projPath


def enterInfoPrepare():  # 输入姓名函数
    name = input('name:')
    if name == 'quit':
        cv2.destroyAllWindows()
        sys.exit(0)
    path = projPath + '/img/' + name
    if not os.path.exists(path):
        os.mkdir(path)
    return path, name
