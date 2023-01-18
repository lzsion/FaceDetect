import numpy as np


def getEuclideanDist(feature_1, feature_2):     # 获得两个向量的欧式距离
    feature_1 = np.array(feature_1)
    feature_2 = np.array(feature_2)
    dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
    return dist
