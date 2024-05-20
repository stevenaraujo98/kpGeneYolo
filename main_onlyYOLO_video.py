from ultralytics import YOLO

model = YOLO('yolov8x-pose-p6.pt')

# results = model('https://ultralytics.com/images/bus.jpg', show=True, conf=0.5, save=True)
# results = model('video.avi', show=True, conf=0.5, save=True)
# results = model(source=0, show=True, conf=0.5, save=True)
# results = model("./StereoVision DataBase/Laptop/waiter/2D/videos/Config1/00_25_49_24_02_2024_VID_LEFT.avi", tracker="bytetrack.yaml", show=True, conf=0.5, save=True, save_txt=True)
# results = model("./StereoVision DataBase/Laptop/waiter/2D/videos/Config1/00_25_49_24_02_2024_VID_RIGHT.avi", tracker="bytetrack.yaml", show=True, conf=0.5, save=True, save_txt=True)
# results = model("./StereoVision DataBase/Laptop/waiter/2D/videos/Config1/00_27_12_24_02_2024_VID_LEFT.avi", tracker="bytetrack.yaml", show=True, conf=0.5, save=True, save_txt=True)
# results = model("./StereoVision DataBase/Laptop/waiter/2D/videos/Config1/00_27_12_24_02_2024_VID_RIGHT.avi", tracker="bytetrack.yaml", show=True, conf=0.5, save=True, save_txt=True)
# results = model("./StereoVision DataBase/Laptop/waiter/2D/videos/Config1/16_35_42_26_02_2024_VID_LEFT.avi", tracker="bytetrack.yaml", show=True, conf=0.5, save=True, save_txt=True)
# results = model("./StereoVision DataBase/Laptop/waiter/2D/videos/Config1/16_35_42_26_02_2024_VID_RIGHT.avi", tracker="bytetrack.yaml", show=True, conf=0.5, save=True, save_txt=True)

# results = model("./StereoVision DataBase/Jetson/waiter/2D/videos/Config2/17_47_25_18_02_2024_VIDEO_LEFT.avi", tracker="bytetrack.yaml", show=True, conf=0.5, save=True, save_txt=True)
# results = model("./StereoVision DataBase/Jetson/waiter/2D/videos/Config2/17_47_25_18_02_2024_VIDEO_RIGHT.avi", tracker="bytetrack.yaml", show=True, conf=0.5, save=True, save_txt=True)
# results = model("./StereoVision DataBase/Jetson/waiter/2D/videos/Config2/17_50_55_18_02_2024_VIDEO_LEFT.avi", tracker="bytetrack.yaml", show=True, conf=0.5, save=True, save_txt=True)
# results = model("./StereoVision DataBase/Jetson/waiter/2D/videos/Config2/17_50_55_18_02_2024_VIDEO_RIGHT.avi", tracker="bytetrack.yaml", show=True, conf=0.5, save=True, save_txt=True)
# results = model("./StereoVision DataBase/Jetson/waiter/2D/videos/Config2/17_54_03_18_02_2024_VIDEO_LEFT.avi", tracker="bytetrack.yaml", show=True, conf=0.5, save=True, save_txt=True)
# results = model("./StereoVision DataBase/Jetson/waiter/2D/videos/Config2/17_54_03_18_02_2024_VIDEO_RIGHT.avi", tracker="bytetrack.yaml", show=True, conf=0.5, save=True, save_txt=True)

# results = model("./StereoVision DataBase/Jetson/rosmasterx3plus/2D/videos/Config2/17_16_14_18_02_2024_VIDEO_LEFT.avi", tracker="bytetrack.yaml", show=True, conf=0.5, save=True, save_txt=True)
# results = model("./StereoVision DataBase/Jetson/rosmasterx3plus/2D/videos/Config2/17_16_14_18_02_2024_VIDEO_RIGHT.avi", tracker="bytetrack.yaml", show=True, conf=0.5, save=True, save_txt=True)
# results = model("./StereoVision DataBase/Jetson/rosmasterx3plus/2D/videos/Config2/17_25_17_18_02_2024_VIDEO_LEFT.avi", tracker="bytetrack.yaml", show=True, conf=0.5, save=True, save_txt=True)
# results = model("./StereoVision DataBase/Jetson/rosmasterx3plus/2D/videos/Config2/17_25_17_18_02_2024_VIDEO_RIGHT.avi", tracker="bytetrack.yaml", show=True, conf=0.5, save=True, save_txt=True)
# results = model("./StereoVision DataBase/Jetson/rosmasterx3plus/2D/videos/Config2/17_29_54_18_02_2024_VIDEO_LEFT.avi", tracker="bytetrack.yaml", show=True, conf=0.5, save=True, save_txt=True)
# results = model("./StereoVision DataBase/Jetson/rosmasterx3plus/2D/videos/Config2/17_29_54_18_02_2024_VIDEO_RIGHT.avi", tracker="bytetrack.yaml", show=True, conf=0.5, save=True, save_txt=True)

# results = model("./StereoVision DataBase/Laptop/rosmasterx3plus/2D/videos/Config1/00_40_47_24_02_2024_VID_LEFT.avi", tracker="bytetrack.yaml", show=True, conf=0.5, save=True, save_txt=True)
# results = model("./StereoVision DataBase/Laptop/rosmasterx3plus/2D/videos/Config1/00_40_47_24_02_2024_VID_RIGHT.avi", tracker="bytetrack.yaml", show=True, conf=0.5, save=True, save_txt=True)
# results = model("./StereoVision DataBase/Laptop/rosmasterx3plus/2D/videos/Config1/00_46_35_24_02_2024_VID_LEFT.avi", tracker="bytetrack.yaml", show=True, conf=0.5, save=True, save_txt=True)
# results = model("./StereoVision DataBase/Laptop/rosmasterx3plus/2D/videos/Config1/00_46_35_24_02_2024_VID_RIGHT.avi", tracker="bytetrack.yaml", show=True, conf=0.5, save=True, save_txt=True)
# results = model("./StereoVision DataBase/Laptop/rosmasterx3plus/2D/videos/Config1/00_52_09_24_02_2024_VID_LEFT.avi", tracker="bytetrack.yaml", show=True, conf=0.5, save=True, save_txt=True)
# results = model("./StereoVision DataBase/Laptop/rosmasterx3plus/2D/videos/Config1/00_52_09_24_02_2024_VID_RIGHT.avi", tracker="bytetrack.yaml", show=True, conf=0.5, save=True, save_txt=True)


# results = model("./Videos/16_35_42_26_02_2024_VID_LEFT.avi", tracker="bytetrack.yaml", show=True, conf=0.5, save=True, save_txt=True)
# results = model("./Videos/16_35_42_26_02_2024_VID_RIGHT.avi", tracker="bytetrack.yaml", show=True, conf=0.5, save=True, save_txt=True)


list_name = [
    # "14_42_36_12_04_2024_VID_",
    # "14_46_06_12_04_2024_VID_",
    "14_49_32_12_04_2024_VID_",
    "14_52_33_12_04_2024_VID_",
    "14_55_02_12_04_2024_VID_",
    "14_57_35_12_04_2024_VID_",
    "15_00_24_12_04_2024_VID_",
    "15_02_56_12_04_2024_VID_",
    "15_05_29_12_04_2024_VID_",
    "15_08_51_12_04_2024_VID_",
]

for name in list_name:
    model = YOLO('yolov8x-pose-p6.pt')
    # name = "14_46_06_12_04_2024_VID_"
    print("Star => ./StereoVision DataBase/Laptop/stretch/" + name + "LEFT_calibrated.avi")
    results = model("./StereoVision DataBase/Laptop/stretch/" + name + "LEFT_calibrated.avi", tracker="bytetrack.yaml", show=True, conf=0.5, save=True, save_txt=True)
    print(results)
    print("END --------------------------")

    model = YOLO('yolov8x-pose-p6.pt')
    print("Star => ./StereoVision DataBase/Laptop/stretch/" + name + "RIGHT_calibrated.avi")
    results = model("./StereoVision DataBase/Laptop/stretch/" + name + "RIGHT_calibrated.avi", tracker="bytetrack.yaml", show=True, conf=0.5, save=True, save_txt=True)
    print(results)
    print("END --------------------------")


print(results)
#model.predict(frame)

# boxes = results.boxes  # Boxes object for bounding box outputs
# masks = results.masks  # Masks object for segmentation masks outputs
# keypoints = results.keypoints  # Keypoints object for pose outputs
# probs = results.probs  # Probs object for classification outputs
# print(boxes, masks, keypoints, probs)