import csv
import cv2
import numpy as np
from PIL import ImageGrab

from FaceDetect.enterInfoPrepare import enterInfoPrepare
from FaceDetect.faceDetect import faceDetect
from FaceDetect.initPath import namesList, featuresKnownArray, featuresPath, namesPath
from FaceDetect.saveFaceDetect import saveFaceDetect
from FaceDetect.settings import cameraIndex


def startDetect():  # 开始检测函数
    camera = cv2.VideoCapture(cameraIndex)  # 摄像头
    useScreen = True  # 默认用屏幕
    index = 1  # 录入时截取图片编号
    maxIndex = 100  # 录入时截取图片数量
    saveFlag = False  # 录入标志
    inputName = False  # 输入姓名标志
    featuresList = []  # 人脸特征向量列表
    while True:
        if useScreen:
            screen = ImageGrab.grab()  # 截取屏幕
            screenImg = cv2.cvtColor(np.asarray(screen), cv2.COLOR_RGB2BGR)  # 转换图像为RGB
            frame = cv2.resize(screenImg, dsize=(960, 540))  # 每一帧
        else:
            retFlag, frame = camera.read()  # 获取摄像头
        k = cv2.waitKey(1) & 0xFF  # 键盘按键

        if k == ord('q'):  # 按q退出
            break

        if k == ord('1'):  # 按1用屏幕
            useScreen = True
        elif k == ord('2'):  # 按2用摄像头
            useScreen = False

        if k == ord('n') and saveFlag == False:  # 新建人物
            cv2.destroyAllWindows()     # 暂时关闭窗口
            print('new people')
            path, name = enterInfoPrepare()     # 调用输入信息函数 返回路径和姓名
            inputName = True    # 变更输入姓名标志
            namesList.append(name)  # 将输入的姓名加入姓名列表
        if inputName:   # 如果已经新建了信息
            if k == ord('s'):  # 按s开始/继续保存图像
                featuresList = []
                saveFlag = True
                cv2.destroyAllWindows()
                print('start')
            elif k == ord('d'):  # 按d暂停保存图像
                saveFlag = False
                print('stop')
            elif k == ord('r'):  # 按r重新开始
                print('reset')
                index = 1
                featuresList = []
                saveFlag = False
            if index > maxIndex or k == ord('f'):  # 录入完毕 按f直接结束录入
                featuresArray = np.array(featuresList)
                featuresMean = np.mean(featuresArray, axis=0)
                featuresKnownArray.append(featuresMean)
                with open(featuresPath, "a+", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(featuresMean)
                with open(namesPath, "a+", newline="", encoding='utf-8-sig') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([name])
                index = 0
                saveFlag = False
                inputName = False
                featuresList = []
                cv2.destroyAllWindows()
                print('finished')
                print('insert faces ' + str(index))

        if saveFlag:    # 录入信息
            # 调用录入人脸信息函数 得到人脸特征向量
            index, faceFeatures = saveFaceDetect(frame, index, path, name)
            if len(faceFeatures) > 0:   # 如果有人脸信息
                featuresList.append(faceFeatures)
        else:   # 正常检测
            faceDetect(frame)

    camera.release()    # 释放摄像头
    cv2.destroyAllWindows()     # 关闭窗口
