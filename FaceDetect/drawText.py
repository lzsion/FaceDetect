import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from FaceDetect.settings import fontPath

font = ImageFont.truetype(fontPath, 14)


def drawText(img, text, posi, color):   # 在图像上绘制文本信息
    imgPil = Image.fromarray(img)
    draw = ImageDraw.Draw(imgPil)
    draw.text(posi, text, color, font)
    img = np.array(imgPil)
    return img
