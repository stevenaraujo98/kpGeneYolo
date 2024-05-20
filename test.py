from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
from calculate import get_resolution

# Create a new YOLO model from scratch (crear desde cero)
# model = YOLO('yolov8n.yaml')

# Load a pretrained YOLO model (recommended for training)
# model = YOLO('yolov8n.pt')
model = YOLO('yolov8m-pose.pt')

# # Train the model using the 'coco128.yaml' dataset for 3 epochs
# results = model.train(data='coco128.yaml', epochs=3)

# # Evaluate the model's performance on the validation set
# results = model.val()

# Perform object detection on an image using the model
# results = model('https://ultralytics.com/images/bus.jpg', show=True, conf=0.3, save=True)
# results = model(source=0, show=True, conf=0.3, save=True)
#
'''
class LivePlot3d:
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.keep_loop = True
        self.ax.set_ylim(-1, 1)
        self.ax.set_xlim(-3, 3)
        self.ax.set_zlim(-5, 5)

    def plot(self, x, y, z, marker="o"):
        self.ax.scatter(x, y, z)
        plt.draw()
        plt.pause(0.1)

    def clean(self):
        self.ax.cla()

plot3d = LivePlot3d()

def find_depth_from_disparities(right_points, left_points,
                                baseline=16, f_pixel=829.4):
    """compute depth from a list of x coordinates
    Parameters:
        left_points(np.ndarray): 1xn or nx1 numpy array containing keypoints x coordinates as viewed from the left camera
        right_points(np.ndarray): 1xn or nx1 numpy array containing keypoints x coordinates as viwed from the right camera
        baseline (float): distance between cameras
        f_pixel (float): focal length of the stereo vision system in pixel units
    """

    x_right = np.array(right_points)
    x_left = np.array(left_points)

    # CALCULATE THE DISPARITY:
    # Displacement between left and right frames [pixels]
    disparity = np.abs(x_left-x_right)

    # CALCULATE DEPTH z:
    zDepth = (baseline*f_pixel)/disparity  # Depth in [cm]

    return np.mean(zDepth)

def face3d(face_left, face_right, baseline, f_px, center_left):
    assert len(face_left) == len(face_right)

    z = [find_depth_from_disparities(
        [x1[0]], [x2[0]], baseline, f_px) for x1, x2 in zip(face_left, face_right)]

    x = (face_left[:, 0] - center_left[0])*z/f_px
    y = (face_left[:, 1] - center_left[1])*z/f_px
    #  for p_left, p_right in zip(face_left, face_right):
    #      depth = find_depth_from_disparities(
    #          [p_left[0]], [p_right[0]], baseline, f_px)
    #      z.append(depth)

    return x, y, z

def pointCloud(leftKpts, rightKpts):
    nubes = []
    plot3d.clean()
    for leftK, rightK in zip(leftKpts, rightKpts):
        points =face3d(leftK, rightK, baseline, fpx, center)
        print(keypointsL, keypointsR)
        print(points)
        plot3d.plot(*points)
'''

import matplotlib.pyplot as plt
import numpy as np

figure = None
def setup_plot():
    global figure
    if figure is not None:
        return figure
    print("setup plot")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_ylim(-2000, 2000)
    ax.set_xlim(-3000, 3000)
    ax.set_zlim(-5000, 5000)
    figure = fig, ax
    return fig, ax

def plot_3d(x, y, z, ax, marker="o"):
    ax.scatter(x, y, z)
    plt.draw()
    plt.pause(0.1)

def clean_plot(ax):
    ax.cla()
    ax.set_ylim(-1000, 2000)
    ax.set_xlim(-3000, 3000)
    ax.set_zlim(0, 10000)


def live_plot_3d(left_kpts, right_kpts, baseline, f_px, center):
    print("live plot")
    fig, ax = setup_plot()
    clean_plot(ax)    
    for left_k, right_k in zip(left_kpts, right_kpts):
        points = face_3d(left_k, right_k, baseline, f_px, center)
        plot_3d(*points, ax)

def find_depth_from_disparities(right_points, left_points, baseline, f_pixel):
    x_right = np.array(right_points)
    x_left = np.array(left_points)
    disparity = np.abs(x_left - x_right)
    z_depth = (baseline * f_pixel) / disparity
    return np.mean(z_depth)

def face_3d(face_left, face_right, baseline, f_px, center_left):
    assert len(face_left) == len(face_right)

    z = [find_depth_from_disparities(
        [x1[0]], [x2[0]], baseline, f_px) for x1, x2 in zip(face_left, face_right)]

    x = (face_left[:, 0] - center_left[0]) * z / f_px
    y = (face_left[:, 1] - center_left[1]) * z / f_px

    return x, y, z

def point_cloud(left_kpts, right_kpts, baseline, f_px, center):
    print("point cloud")
    live_plot_3d(left_kpts, right_kpts, baseline, f_px, center)


def getStereoRectifier(calib_file):
    """Build rectifier from stereo map file
    Parameters:
        calib_file (str): file name of the stereo map file generated with calibration procedure
    Returns:
        (np.ndarray, np.ndarray)->(np.ndarray,np.ndarray) rectify function takes 2 unrectified images and returns those images calibrated
    """

    # Camera parameters to undistort and rectify images
    cv_file = cv2.FileStorage()
    cv_file.open(calib_file, cv2.FileStorage_READ)

    stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
    stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
    stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
    stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()

    def undistortRectify(frameL, frameR):

        # Undistort and rectify images
        undistortedL = cv2.remap(frameL, stereoMapL_x, stereoMapL_y,
                                 cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        undistortedR = cv2.remap(frameR, stereoMapR_x, stereoMapR_y,
                                 cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

        return undistortedL, undistortedR

    return undistortRectify

# Ejemplo de uso:
left_keypoints = np.random.rand(10, 2)  # Ejemplo de keypoints izquierdos
right_keypoints = np.random.rand(10, 2)  # Ejemplo de keypoints derechos
baseline = 58
f_px = 1047.7021754075784
center = (914.4276428222656,
            532.6548118591309)



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


capL=cv2.VideoCapture('./predictionYolo/predict39/16_35_42_26_02_2024_VID_LEFT.avi')
capR=cv2.VideoCapture('./predictionYolo/predict40/16_35_42_26_02_2024_VID_RIGHT.avi')

#rectfier = getStereoRectifier('./calibration/stereoMap.yml')
#cap.open()
while(capR.isOpened() and capL.isOpened()):
    print("running")
    ret,frameL = capL.read()
    retR,frameR = capR.read()

    frame_combined = frameL/2 + frameR/2
    frame_combined = frame_combined.astype(np.uint8)
    cv2.imshow("comnb", frame_combined)
    h = frameR.shape[0]
    w = frameR.shape[1]

    if(not ret or not retR):
        print("Failed to read frames")

    cv2.imshow("jhljhljh",frameL)
    key = cv2.waitKey(1)
    if key == ord('q'):
        # Close
        break
    
    # Predict
    resultL = model.predict(frameL)
    resultR = model.predict(frameR)

    keypointsL = np.array(resultL[0].keypoints.xy.cpu())
    keypointsR = np.array(resultR[0].keypoints.xy.cpu())
    
    print("LEN", len(keypointsL))
    print(keypointsL)

    if (len(keypointsL)<=1 or len(keypointsR)<=1):continue
    # keypointsL = list(keypointsL).sort(key=lambda kpts:np.mean(kpts[:,0]))
    # keypointsR = list(keypointsR).sort(key=lambda kpts:np.mean(kpts[:,0]))

    # keypointsL = np.array(keypointsL)
    # keypointsR = np.array(keypointsR)

    # print(result[0].keypoints.xy)
    # print(result[0].keypoints.xy.shape)
    try:
        print(get_resolution(keypointsL, keypointsR))

        graph_circles(keypointsL, w, h, frameL)
        graph_circles(keypointsR, w, h, frameR)
    except:
        print("Error")
        continue

    cv2.waitKey(0)
    cv2.imshow('LEFT',frameL)
    cv2.imshow('RIGHT',frameR)


    # pointCloud(keypointsL,keypointsR)
    #keypointsL_sorted = sorted()
    keypointsR_sorted = [keypointsR[index] for index in get_resolution(keypointsL, keypointsR).values()]
    print("sorted")
    point_cloud(keypointsL, keypointsR_sorted, baseline, f_px, center)




#results = model(source="./Videos/00_27_12_24_02_2024_VID_RIGHT.avi", show=True, conf=0.3, save=True)
#model.predict
# Export the model to ONNX format
# success = model.export(format='onnx')
# print(results.count())
capL.release()
capR.release()
cv2.destroyAllWindows()
