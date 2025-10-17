
import cv2
import os

from tqdm import trange

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') 
 
fps = 10
size = (640, 480)

path = './img'
fileList = os.listdir(path)
fileCnt = len(fileList)

resultName =  'res.mp4'
video = cv2.VideoWriter(resultName, fourcc, fps, size, isColor=True)

for i in trange(fileCnt):
    fileName = "step_" + str(i) + ".png"

    img = cv2.imread(path + '/' + fileName)
    # cv2.imshow('img', img)
    video.write(img)
    
video.release()