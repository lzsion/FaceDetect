import cv2

from FaceDetect.borderText import borderText
from FaceDetect.drawText import drawText
from FaceDetect.getEuclideanDist import getEuclideanDist
from FaceDetect.initPath import detector, predictor, faceRec, featuresKnownArray, namesList


def faceDetect(img):  # 正常人脸检测识别函数
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     # 转化为灰度图像
    dets = detector(grayImg, 1)     # 人脸检测器
    namePosiList = []     # 人脸姓名位置列表
    detectNameList = []     # 人脸姓名列表
    if len(dets) > 0:   # 如果检测到了人脸
        featuresCapArray = []
        for i in range(len(dets)):  # 遍历捕获到的图像中所有的人脸
            shape = predictor(img, dets[i])
            featuresCapArray.append(faceRec.compute_face_descriptor(img, shape))    # 获得图像中人脸的特征向量
        for k in range(len(dets)):
            # 让人名跟随在矩形框的下方
            # 确定人名的位置坐标
            # 先默认所有人不认识，是 unknown
            detectNameList.append("unknown")

            # 每个捕获人脸的名字坐标
            namePosiList.append(tuple([dets[k].left(), int(dets[k].bottom())]))

            # 对于某张人脸，遍历所有存储的人脸特征
            e_distance_list = []
            for i in range(len(featuresKnownArray)):
                # 如果 person_X 数据不为空
                if str(featuresKnownArray[i][0]) != '0.0':
                    e_distance_tmp = getEuclideanDist(featuresCapArray[k], featuresKnownArray[i])
                    e_distance_list.append(e_distance_tmp)
                else:
                    # 空数据 person_X
                    e_distance_list.append(999999999)
            # 找出最接近的一个人脸数据是第几个
            similar_person_num = e_distance_list.index(min(e_distance_list))

            # 计算人脸识别特征与数据集特征的欧氏距离
            # 距离小于0.4则标出为可识别人物
            if min(e_distance_list) < 0.4:
                detectNameList[k] = namesList[similar_person_num]
            for kk, d in enumerate(dets):
                # 绘制矩形框
                cv2.rectangle(img, tuple([d.left(), d.top()]), tuple([d.right(), d.bottom()]), (0, 255, 0), 1)

        # 在人脸框下面写人脸名字
        for i in range(len(dets)):
            img = drawText(img, detectNameList[i], namePosiList[i], (0, 0, 255))

    img = borderText(img, len(dets))    # 图像加上旁边信息
    cv2.imshow("detect", img)   # 窗口显示图像
