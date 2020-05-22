
import cv2
import numpy as np
import glob

def write_video():
    img_array = []
    for filename in glob.glob('C:/Users/Wenjin Tao/PycharmProjects/Xuyong-AOI/temp/*.jpg'):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter('mask_inspection1.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

write_video()