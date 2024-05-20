from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt

model = YOLO('yolov8m-pose.pt')

fpx = 1168.3260275144676
baseline = 0.058
center= np.array([
            985.1602172851562,
            536.3686485290527
        ])

cap = cv2.VideoCapture('./Videos/00_27_12_24_02_2024_VID_RIGHT.avi')
capL = cv2.VideoCapture('./Videos/00_27_12_24_02_2024_VID_LEFT.avi')


