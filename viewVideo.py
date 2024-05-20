from ultralytics import YOLO
import numpy as np
import cv2

model = YOLO('yolov8x-pose-p6.pt')

cap = cv2.VideoCapture('./Videos/00_27_12_24_02_2024_VID_RIGHT.avi')
capL = cv2.VideoCapture('./Videos/00_27_12_24_02_2024_VID_LEFT.avi')

while(cap.isOpened() and capL.isOpened()):
    ret,frame = cap.read()
    frame_copy = frame.copy()
    ret,frameR = cap.read()
    key = cv2.waitKey(1)
    if key == ord('q'):
        # Close
        break

    resultL = model.predict(frame)
    # print(frame.shape)
    resultR = model.predict(frameR)
    # print(frameR.shape)

    print(resultL)
    keypointsL = np.array(resultL[0].keypoints.xy.cpu())
    keypointsR = np.array(resultR[0].keypoints.xy.cpu())
    print("Persona", len(keypointsL))

    for person in keypointsL:
        for kp in person:
            pos_x = int(float(kp[0]))
            pos_y = int(float(kp[1]))
            if (pos_x != 0 and pos_y != 0):
                cv2.circle(frame, (pos_x, pos_y), 3, (255,0,255), 3)

    cv2.imshow("",frame_copy)
    cv2.imshow("KP",frame)


    """
    pointCloud(keypointsL,keypointsR)
    """

