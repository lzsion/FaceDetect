import random
import cv2

from FaceDetect.borderText import borderText
from FaceDetect.cv2ImageRW import cv2imwrite
from FaceDetect.drawText import drawText
from FaceDetect.initPath import detector, predictor, faceRec
from FaceDetect.relight import relight


def saveFaceDetect(img, index, path, name):  # 获得人脸特征向量
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     # 转化为灰度图像
    dets = detector(grayImg, 1)     # 人脸检测器
    faceFeatures = []   # 人脸特征向量
    face = []
    for i, d in enumerate(dets):
        # 人脸区域
        x1 = d.top() if d.top() > 0 else 0
        y1 = d.bottom() if d.bottom() > 0 else 0
        x2 = d.left() if d.left() > 0 else 0
        y2 = d.right() if d.right() > 0 else 0
        face = img[x1:y1, x2:y2].copy()
        # 调整图片的对比度与亮度   对比度与亮度值都取随机数增加样本的多样性
        face = relight(face, random.uniform(0.5, 1.5), random.randint(-50, 50))
        faceDet = detector(face, 1)
        if len(faceDet) == 1:  # 仅检测到一张人脸
            faceShape = predictor(face, faceDet[0])     # 68点人脸特征
            faceFeatures = faceRec.compute_face_descriptor(face, faceShape)     # 68点人脸特征转化为向量
            cv2imwrite(path + '/' + name + '_' + str(index) + '.png', face)     # 保存图片
            text = '正在录入: ' + str(index)    # 窗口显示文本
            img = drawText(img, text, (10, 10), (0, 0, 255))    # 绘制文本
            index += 1
        shape = predictor(img, dets[0])
        for pt in shape.parts():    # 绘制68个特征点
            ptPosi = (pt.x, pt.y)
            cv2.circle(img, ptPosi, 1, (0, 255, 0), 1)
        cv2.rectangle(img, (x2, x1), (y2, y1), color=(0, 255, 0), thickness=1)

    img = borderText(img, len(dets))    # 图像加上旁边信息
    cv2.imshow('detect', img)   # 窗口显示图像
    return index, faceFeatures
