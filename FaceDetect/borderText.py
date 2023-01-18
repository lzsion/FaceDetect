import cv2

from FaceDetect.drawText import drawText


def borderText(img, faceNum):   # 窗口旁边信息
    img = cv2.copyMakeBorder(img, 0, 0, 130, 0, cv2.BORDER_CONSTANT)
    text = ["人脸识别系统",
            "-" * 18,
            "人脸数量: " + str(faceNum),
            "-" * 18,
            "按'1'/'2'切换镜头",
            "按'n'添加人脸",
            "按's'开始录入",
            "按'd'暂停",
            "按'r'重新录入",
            "按'f'结束录入",
            "按'q'退出",
            "-" * 18]
            # "Editor: LZS"]
    y = 10
    for eachText in text:
        img = drawText(img, eachText, (10, y), (255, 255, 255))
        y += 20
    return img
