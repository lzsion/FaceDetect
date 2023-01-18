# 基于dlib库的人脸检测和人脸识别

## 前言

此项目为课程作业，原本使用opencv自带的人脸检测器来实现人脸检测，参考[b站opencv人脸识别视频](https://www.bilibili.com/video/BV1Lq4y1Z7dm/)。

但是实际程序识别精度较低，经常出现误识别和漏识别的问题，查找资料后选择使用dlib库，参考博客链接附在最后。

此项目已经作为课程作业提交，时间有限，程序界面比较简陋，仍有未修复的bug，但不影响正常使用。

此程序由python语言编写，运用到的模型有

已经训练好的dlib人脸检测器

> dlib_face_recognition_resnet_model_v1.dat

已经训练好的68人脸特征点检测器

>shape_predictor_68_face_landmarks.dat

以上模型均从官网下载。

此程序窗口界面由opencv生成，对于显示在界面上的中文无法显示，故调用中文字体库解决，参考博客[opencv在图片上绘制中文](https://blog.csdn.net/sinat_29957455/article/details/105069917)。

用dilb库进行人脸识别，在精度上比opencv自带的人脸识别要准确，但所需的算力更大，在进行实时的识别时，显示的帧率较低。为了改善此问题，使用了GPU版本的dlib，用CUDA模块加速，在实际的显示帧率上有所改善，参考博客[dlib的GPU加速](https://blog.csdn.net/luckyfairy17/article/details/83855739)。虽然使用了GPU加速了dlib，但由于电脑算力不足，程序优化不够，程序显示图像的帧率较低。

## 程序示例

以下图片为[演示视频](https://lzs-imgs.oss-cn-hangzhou.aliyuncs.com/face-detect/example_video_1.mp4)的截图。

人脸识别示例如下，已经录入了丁真珍珠的人脸数据，程序运行时，实时抓取电脑主屏幕图像，与人脸数据库中的比对，能够正确识别出丁真珍珠，并在框选的人脸下方标记人物姓名。程序默认抓取电脑主屏幕，也可以切换为电脑摄像头。

![人脸识别示例1](https://lzs-imgs.oss-cn-hangzhou.aliyuncs.com/face-detect/face-detect-1.png)

人脸录入示例如下，其中左上角为电脑主屏幕，右下角为此程序的窗口。

为使人脸录入有效，需要保证人脸录入环境仅有一张人脸，这里选择b站up主[罗翔说刑法](https://space.bilibili.com/517327498)的单人视频为录入样例。

按"n"键添加人物，在终端输入姓名，回车确认，再按"s"键开始录入，录入时抓取屏幕中的人脸，获取人的68点人脸特征点，默认抓取100张人脸照片，获得100张人脸对应的68点特征点，求平均值得到该人物的68点人脸特征点。

录入期间可以按照程序左侧提示暂停录入，重新录入，提前结束录入，~~这些功能可能存在bug~~。

![人脸录入示例](https://lzs-imgs.oss-cn-hangzhou.aliyuncs.com/face-detect/face-detect-2.png)

抓取100张照片后，录入程序自动跳转至识别程序，如下图。

![人脸识别示例2](https://lzs-imgs.oss-cn-hangzhou.aliyuncs.com/face-detect/face-detect-3.png)

## 代码介绍

程序运行时，加载"names_all.csv"和"features_all.csv"两个csv文件，分别为已有的人物名字以及其对应的人脸特征向量，并加载所需的检测器，封装至"initPath.py"。

程序使用opencv窗口显示，默认获取电脑屏幕图像，可以在后续通过键盘按键调整为获取摄像头图像。人脸信息加载完成后，运行人脸识别程序(“faceDetect.py”中的faceDetect函数)。

人脸录入时，调用人脸录入程序(“saveFaceDetect.py”中的saveFaceDetect函数)。为了方便录入，本程序通过截取实时视频图片的方法，保证摄像头或者屏幕只有一个人脸，通过摄像头或者屏幕的每一帧的图片，用dlib的人脸正脸识别检测器，选出人脸部分，再随机调整人脸部分的曝光度和饱和度(“relight.py”中的relight函数)，使数据集更具有普适性。然后用68点人脸关键点检测器得到每一帧的人脸特征向量，取100张人脸照片对应的人脸特征向量平均值，得到最终的人脸特征向量。

调用68点人脸特征点检测器得到每一张人脸照片的人脸特征向量，求所有人脸特征向量的平均值作为此人的人脸特征向量，并与输入的姓名绑定，并写入对应的csv文件中。

程序启动后，默认状态为人脸识别。人脸识别时，使用dlib的人脸正脸识别检测器找到人脸，遍历图片中的所有人脸，用68点人脸特征点检测器获取对应的人脸特征向量，分别与已知的人脸特征向量计算欧式距离(“getEuclideanDist.py”中的getEuclideanDist函数)，设定阈值为0.4，如果欧式距离小于0.4，则可以认为此人脸和已有的人脸对应，并在其下方显示人名。为显示中文，需要调用PIL库绘制，将绘制文本封装至“drawText.py”中的drawText函数。

## 参考链接

[b站opencv人脸识别视频](https://www.bilibili.com/video/BV1Lq4y1Z7dm/)

[dlib人脸检测](https://blog.csdn.net/liuxiao214/article/details/83411820)

[dlib的68个特征点识别](https://www.cnblogs.com/qiynet/p/12801601.html)

[计算欧式距离区别不同人脸(a)](https://blog.csdn.net/qq_22764813/article/details/102974988)

[计算欧式距离区别不同人脸(b)](https://blog.csdn.net/u012505617/article/details/89191158)

[opencv在图片上绘制中文](https://blog.csdn.net/sinat_29957455/article/details/105069917)

[dlib的GPU加速](https://blog.csdn.net/luckyfairy17/article/details/83855739)
