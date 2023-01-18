import dlib
import pandas as pd

from FaceDetect.settings import featuresPath, predPath, recPath, namesPath

detector = dlib.get_frontal_face_detector()  # 人脸检测器

predictor = dlib.shape_predictor(predPath)  # 68点特征识别检测器

faceRec = dlib.face_recognition_model_v1(recPath)   # 调用人脸识别模型

# 加载已有人脸特征向量和人脸信息
csvRead = pd.read_csv(featuresPath, header=None)
featuresKnownArray = []  # 所有录入的人脸特征向量
for i in range(csvRead.shape[0]):
    features_someone_arr = []
    for j in range(0, len(csvRead.iloc[i, :])):
        features_someone_arr.append(csvRead.iloc[i, :][j])
    featuresKnownArray.append(features_someone_arr)
print("数据库内人脸数: ", len(featuresKnownArray))

nameRead = pd.read_csv(namesPath, header=None, encoding='utf-8-sig')
namesList = []  # 录入的人脸对应名字
for i in range(nameRead.shape[0]):
    namesList.append(nameRead.iloc[i, :][0])
print(namesList)
