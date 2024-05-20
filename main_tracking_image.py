from ultralytics import YOLO
import cv2
import numpy as np
import random
import glob
from calculate import get_resolution
import os

list_colors = [(255,0,255), (0, 255, 255), (255, 0, 0), (0, 0, 0), (255, 255, 0), (205, 92, 92), (255, 0, 255), (0, 128, 128), (128, 0, 0), (128, 128, 0), (128, 128, 128)]

def graph_circles_by_person(array_pts, frame):
    num_person = 0
    for pt in array_pts:
        pos_x = int(pt[0])
        pos_y = int(pt[1])
        if (pos_x != 0 and pos_y != 0):
            # print(pos_x, pos_y)
            cv2.circle(frame, (pos_x, pos_y), 3, list_colors[num_person], 3)
    num_person+=1

def graph_circles(array_pts, frame):
    num_person = 0
    for person in array_pts:
        for pt in person:
            pos_x = int(pt[0])
            pos_y = int(pt[1])
            if (pos_x != 0 and pos_y != 0):
                # print(pos_x, pos_y)
                cv2.circle(frame, (pos_x, pos_y), 2, list_colors[num_person], 2)
        num_person+=1

def graph_box(box, img):
    box = np.array(box.data.cpu())[0]
    print((box[0], box[1]), (box[2], box[3]), box[4])
    box = box.astype(int)
    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)


# model = YOLO('yolov8m-pose.pt')
model = YOLO('yolov8x-pose-p6.pt')

"""
img = cv2.imread("./bus.jpg")

# if you want all classes
yolo_classes = list(model.names.values())
classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]

conf = 0.65

results = model.predict(img, conf=conf, classes=[0])
colors = [random.choices(range(256), k=3) for _ in classes_ids]

# colors = [random.choices(range(256), k=3) for _ in classes_ids]


def graph_box(box, img):
    box = np.array(box.data.cpu())[0]
    print((box[0], box[1]), (box[2], box[3]), box[4])
    box = box.astype(int)
    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

for result in results:
    for keypoints, box in zip(result.keypoints.xy.cpu(), result.boxes):
        graph_circles_by_person(np.array(keypoints), img)
        
        graph_box(box, img)
    

cv2.imshow("Image", img)
cv2.waitKey(0)

cv2.destroyAllWindows()
"""






# if you want all classes
yolo_classes = list(model.names.values())
classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]
conf = 0.5

PATH_SAVE = "./StereoVision DataBase/Laptop/"
robot_selected = "rosmasterx3plus"
configNum = "Config1"
path_save_final = PATH_SAVE + robot_selected + "/2D/"
distance = "250 y 500"

# lista de los nombres de las imagenes completos
image_files = glob.glob(path_save_final + "images/" + distance + "/calibrated/integradora/*_LEFT_CALIB.jpg")
# image_files = glob.glob(path_save_final + "images/" + distance + "/calibrated/re_calibration2/*_LEFT_CALIB.jpg")
for image_file in image_files:
    # acortar el nombre de la imagen menos la extension y el tipo
    name = image_file[84 + 6 :-15]
    # name = image_file[88 + 6:-15]
    frameL = cv2.imread(path_save_final + "images/" + distance + "/calibrated/integradora/" + name + "_LEFT_CALIB.jpg")
    # frameL = cv2.imread(path_save_final + "images/" + distance + "/calibrated/re_calibration2/" + name + "_LEFT_CALIB.jpg")
    frameR = cv2.imread(path_save_final + "images/" + distance + "/calibrated/integradora/" + name + "_RIGHT_CALIB.jpg")
    # frameR = cv2.imread(path_save_final + "images/" + distance + "/calibrated/re_calibration2/" + name + "_RIGHT_CALIB.jpg")

    h = frameR.shape[0]
    w = frameR.shape[1]

    resultL = model.predict(frameL, conf=conf, classes=[0])
    resultR = model.predict(frameR , conf=conf, classes=[0])
    
    keypointsL = np.array(resultL[0].keypoints.xy.cpu())
    keypointsR = np.array(resultR[0].keypoints.xy.cpu())
    keypointsR_sorted = []
    
    print("LEN", len(keypointsL))
    # print(keypointsL)

    # if (len(keypointsL)<=1 or len(keypointsR)<=1):continue
    
    try:
        # get_resolution(keypointsL, keypointsR)
        # quitar el value al guardar +--+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        keypointsR_sorted = [keypointsR[index] for index in get_resolution(keypointsL, keypointsR).values()]
        keypointsR_sorted = np.array(keypointsR_sorted)

        graph_circles(keypointsL, frameL)
        graph_circles(keypointsR_sorted, frameR)
    except Exception as e:
        print(e, "No hay deteccion de keypoints")
        pass

    
    
    # Crear las carpetas para guardar los keypoints
    os.mkdir("results/Laptop/" + robot_selected + "/" + distance + "/" + name + "_LEFT")
    os.mkdir("results/Laptop/" + robot_selected + "/" + distance + "/" + name + "_RIGHT")
    # Gurdar los keypoints en un archivo de texto
    archivo_f_l = open("results/Laptop/" + robot_selected + "/" + distance + "/" + name + "_LEFT/" + name + "_LEFT_1.txt", "w")
    archivo_f_r = open("results/Laptop/" + robot_selected + "/" + distance + "/" + name + "_RIGHT/" + name + "_RIGHT_1.txt", "w")
    # archivo_f = open("results/" + name_video + "/frame_" + str(frame_num) +  ".txt", "w")
    archivo_f_l.write(str(keypointsL))
    archivo_f_r.write(str(keypointsR_sorted))
    archivo_f_l.close()
    archivo_f_r.close()


    # Guardar imagenes con keypoints
    cv2.imwrite("results/Laptop/" + robot_selected + "/" + distance + "/" + name + "_LEFT.jpg", frameL)
    cv2.imwrite("results/Laptop/" + robot_selected + "/" + distance + "/" + name + "_RIGHT.jpg", frameR)
    """
    cv2.imshow('LEFT',frameL)
    cv2.imshow('RIGHT',frameR)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """