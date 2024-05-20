from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
from calculate import get_resolution
import os

# Create a new YOLO model from scratch (crear desde cero)
# model = YOLO('yolov8n.yaml')

# Load a pretrained YOLO model (recommended for training)
# model = YOLO('yolov8n.pt')
# model = YOLO('yolov8m-pose.pt')
model = YOLO('yolov8x-pose-p6.pt')


# # Train the model using the 'coco128.yaml' dataset for 3 epochs
# results = model.train(data='coco128.yaml', epochs=3)

# # Evaluate the model's performance on the validation set
# results = model.val()

# Perform object detection on an image using the model
# results = model('https://ultralytics.com/images/bus.jpg', show=True, conf=0.3, save=True)
# results = model(source=0, show=True, conf=0.3, save=True)


list_colors = [(255,0,255), (0, 255, 255), (255, 0, 0), (0, 0, 0), (255, 255, 0), (205, 92, 92), (255, 0, 255), (0, 128, 128), (128, 0, 0), (128, 128, 0), (128, 128, 128)]

def graph_circles(array_pts, w, h, frame):
    num_person = 0
    for person in array_pts:
        for pt in person:
            pos_x = int(pt[0])
            pos_y = int(pt[1])
            if (pos_x != 0 and pos_y != 0):
                # print(pos_x, pos_y)
                cv2.circle(frame, (pos_x, pos_y), 3, list_colors[num_person], 3)
        num_person+=1

# Jetson waiters
# name_L = "17_47_25_18_02_2024_VIDEO_LEFT"
# name_R = "17_47_25_18_02_2024_VIDEO_RIGHT"
# name_L = "17_50_55_18_02_2024_VIDEO_LEFT"
# name_R = "17_50_55_18_02_2024_VIDEO_RIGHT"
# name_L = "17_54_03_18_02_2024_VIDEO_LEFT"
# name_R = "17_54_03_18_02_2024_VIDEO_RIGHT"

# capL=cv2.VideoCapture('./StereoVision DataBase/Jetson/waiter/2D/videos/Config2/' + name_L + '.avi')
# capR=cv2.VideoCapture('./StereoVision DataBase/Jetson/waiter/2D/videos/Config2/' + name_R + '.avi')

# Jetson rosmasterx3plus
# name_L = "17_16_14_18_02_2024_VIDEO_LEFT"
# name_R = "17_16_14_18_02_2024_VIDEO_RIGHT"
# name_L = "17_25_17_18_02_2024_VIDEO_LEFT"
# name_R = "17_25_17_18_02_2024_VIDEO_RIGHT"
# name_L = "17_29_54_18_02_2024_VIDEO_LEFT"
# name_R = "17_29_54_18_02_2024_VIDEO_RIGHT"

# capL=cv2.VideoCapture('./StereoVision DataBase/Jetson/rosmasterx3plus/2D/videos/Config2/' + name_L + '.avi')
# capR=cv2.VideoCapture('./StereoVision DataBase/Jetson/rosmasterx3plus/2D/videos/Config2/' + name_R + '.avi')

# Laptop waiter
# name_L = "00_25_49_24_02_2024_VID_LEFT"
# name_R = "00_25_49_24_02_2024_VID_RIGHT"
# name_L = "00_27_12_24_02_2024_VID_LEFT"
# name_R = "00_27_12_24_02_2024_VID_RIGHT"
# name_L = "16_35_42_26_02_2024_VID_LEFT"
# name_R = "16_35_42_26_02_2024_VID_RIGHT"

# capL=cv2.VideoCapture('./StereoVision DataBase/Laptop/waiter/2D/videos/Config1/' + name_L + '.avi')
# capR=cv2.VideoCapture('./StereoVision DataBase/Laptop/waiter/2D/videos/Config1/' + name_R + '.avi')

# Laptop rosmasterx3plus
# name_L = "00_40_47_24_02_2024_VID_LEFT"
# name_R = "00_40_47_24_02_2024_VID_RIGHT"
# name_L = "00_46_35_24_02_2024_VID_LEFT"
# name_R = "00_46_35_24_02_2024_VID_RIGHT"
# name_L = "00_52_09_24_02_2024_VID_LEFT"
# name_R = "00_52_09_24_02_2024_VID_RIGHT"

# capL=cv2.VideoCapture('./StereoVision DataBase/Laptop/rosmasterx3plus/2D/videos/Config1/' + name_L + '.avi')
# capR=cv2.VideoCapture('./StereoVision DataBase/Laptop/rosmasterx3plus/2D/videos/Config1/' + name_R + '.avi')


# name_L = "16_35_42_26_02_2024_VID_LEFT" # 15_59_16_01_04_2024_VIDEO_LEFT
# name_R = "16_35_42_26_02_2024_VID_RIGHT" # 15_59_16_01_04_2024_VIDEO_RIGHT

# capL=cv2.VideoCapture("dataCalibrated/" + name_L + ".avi")
# capR=cv2.VideoCapture("dataCalibrated/" + name_R + ".avi")

"""
os.mkdir("results/" + name_L)
os.mkdir("results/" + name_R)
"""

# Laptop stretch
name = "14_52_33_12_04_2024_VID"

capL=cv2.VideoCapture('./StereoVision DataBase/Laptop/stretch/' + name + '_LEFT_calibrated.avi')
capR=cv2.VideoCapture('./StereoVision DataBase/Laptop/stretch/' + name + '_RIGHT_calibrated.avi')

os.mkdir("results/" + name + "_LEFT")
os.mkdir("results/" + name + "_RIGHT")



# capL=cv2.VideoCapture('./predictionYolo/predict39/16_35_42_26_02_2024_VID_LEFT.avi')
# capR=cv2.VideoCapture('./predictionYolo/predict40/16_35_42_26_02_2024_VID_RIGHT.avi')

save_video = False
frame_num = 1

#rectfier = getStereoRectifier('./calibration/stereoMap.yml')
#cap.open()

while(capR.isOpened() and capL.isOpened()):
    print("running")
    ret,frameL = capL.read()
    retR,frameR = capR.read()


    if(not ret or not retR):
        print("Failed to read frames")
        break

    h = frameR.shape[0]
    w = frameR.shape[1]
    
    # Predict
    resultL = model.predict(frameL, conf=0.5)
    resultR = model.predict(frameR, conf=0.5)

    keypointsL = np.array(resultL[0].keypoints.xy.cpu())
    keypointsR = np.array(resultR[0].keypoints.xy.cpu())
    
    print("LEN", len(keypointsL))
    # print(keypointsL)

    # if (len(keypointsL)<=1 or len(keypointsR)<=1):continue
    
    try:
        # get_resolution(keypointsL, keypointsR)
        graph_circles(keypointsL, w, h, frameL)
        graph_circles(keypointsR, w, h, frameR)
    except:
        print("Error")
        continue

    
    cv2.imshow('LEFT',frameL)
    cv2.imshow('RIGHT',frameR)

    key = cv2.waitKey(1)
    if key == ord('q'):
        # Close
        break


    # pointCloud(keypointsL,keypointsR)
    #keypointsL_sorted = sorted()
    # keypointsR_sorted = [keypointsR[index] for index in get_resolution(keypointsL, keypointsR).values()]
    # print("sorted")


    if save_video:
        salida_L.write(frameL)
        salida_R.write(frameR)
        
        
        # archivo_f_l = open("predictionYolo/predict39/16_35_42_26_02_2024_VID_LEFT/16_35_42_26_02_2024_VID_LEFT_" + str(frame_num) +  ".txt", "w")
        # archivo_f_r = open("predictionYolo/predict40/16_35_42_26_02_2024_VID_RIGHT/16_35_42_26_02_2024_VID_RIGHT_" + str(frame_num) +  ".txt", "w")
        """
        archivo_f_l = open("results/" + name_L + "/" + name_L + "_" + str(frame_num) +  ".txt", "w")
        archivo_f_r = open("results/" + name_R + "/" + name_R + "_" + str(frame_num) +  ".txt", "w")
        """
        archivo_f_l = open("results/" + name + "_LEFT/" + name + "_LEFT_" + str(frame_num) +  ".txt", "w")
        archivo_f_r = open("results/" + name + "_RIGHT/" + name + "_RIGHT_" + str(frame_num) +  ".txt", "w")
        # archivo_f = open("results/" + name_video + "/frame_" + str(frame_num) +  ".txt", "w")
        archivo_f_l.write(str(keypointsL))
        archivo_f_r.write(str(keypointsR))
        archivo_f_l.close()
        archivo_f_r.close()
        frame_num+=1
    else:
        # salida_L = cv2.VideoWriter('predictionYolo/predict39/16_35_42_26_02_2024_VID_LEFT_YOLO.avi',cv2.VideoWriter_fourcc(*'XVID'),20.0,(int(frameL.shape[1]), int(frameL.shape[0])))
        # salida_R = cv2.VideoWriter('predictionYolo/predict40/16_35_42_26_02_2024_VID_RIGHT_YOLO.avi',cv2.VideoWriter_fourcc(*'XVID'),20.0,(int(frameR.shape[1]), int(frameR.shape[0])))
        """
        salida_L = cv2.VideoWriter('results/'  + name_L +  '.avi',cv2.VideoWriter_fourcc(*'XVID'),10.0,(int(frameL.shape[1]), int(frameL.shape[0])))
        salida_R = cv2.VideoWriter('results/' + name_R + '.avi',cv2.VideoWriter_fourcc(*'XVID'),10.0,(int(frameR.shape[1]), int(frameR.shape[0])))
        """
        salida_L = cv2.VideoWriter('results/'  + name +  '_LEFT.avi',cv2.VideoWriter_fourcc(*'XVID'),10.0,(int(frameL.shape[1]), int(frameL.shape[0])))
        salida_R = cv2.VideoWriter('results/' + name + '_RIGHT.avi',cv2.VideoWriter_fourcc(*'XVID'),10.0,(int(frameR.shape[1]), int(frameR.shape[0])))
        # salida = cv2.VideoWriter('results/' + name_video + '.avi',cv2.VideoWriter_fourcc(*'XVID'),20.0,(int(imageToProcess.shape[1]), int(imageToProcess.shape[0])))
        salida_L.write(frameL)
        salida_R.write(frameR)
        
        # archivo_f_l = open("predictionYolo/predict39/16_35_42_26_02_2024_VID_LEFT/16_35_42_26_02_2024_VID_LEFT_" + str(frame_num) +  ".txt", "w")
        # archivo_f_r = open("predictionYolo/predict40/16_35_42_26_02_2024_VID_RIGHT/16_35_42_26_02_2024_VID_RIGHT_" + str(frame_num) +  ".txt", "w")
        """
        archivo_f_l = open("results/" + name_L + "/" + name_L + "_" + str(frame_num) +  ".txt", "w")
        archivo_f_r = open("results/" + name_R + "/" + name_R + "_" + str(frame_num) +  ".txt", "w")
        """
        archivo_f_l = open("results/" + name + "_LEFT/" + name + "_LEFT_" + str(frame_num) +  ".txt", "w")
        archivo_f_r = open("results/" + name + "_RIGHT/" + name + "_RIGHT_" + str(frame_num) +  ".txt", "w")
        # archivo_f = open("results/" + name_video + "/frame_" + str(frame_num) +  ".txt", "w")
        archivo_f_l.write(str(keypointsL))
        archivo_f_r.write(str(keypointsR))
        archivo_f_l.close()
        archivo_f_r.close()
        frame_num+=1

        save_video = not save_video



#results = model(source="./Videos/00_27_12_24_02_2024_VID_RIGHT.avi", show=True, conf=0.3, save=True)
#model.predict
# Export the model to ONNX format
# success = model.export(format='onnx')
# print(results.count())
capL.release()
capR.release()
cv2.destroyAllWindows()
