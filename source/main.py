from sort import *
import numpy as np
import cv2
from ultralytics import YOLO
import copy

# Object detector model
from object_detector import predict

# Worker class
from worker import Worker

# Display
from displays import prepare_display

# from PIL import Image as im

DEBUG_MODE = False

# Output video
SAVE_OUTPUT = False
OUT_PATH = f"../data/debug/result.mp4"

#
VIDEO_NAME = "betonsa_2"
VIDEO_PATH = f"../data/video_data/{VIDEO_NAME}.mp4"

FRAME_COUNT = 1000

# -------------------------------------------------------
# Configurations

CONF_LEVEL = 0.4
# -------------------------------------------------------
# Reading video with cv2
video = cv2.VideoCapture(VIDEO_PATH)

# Objects to detect Yolo
class_IDS = [0]  # default id for person is 0

# loading a YOLO model
model = YOLO("yolov8n.pt")

# geting names from classes
dict_classes = model.model.names

if __name__ == "__main__":
    video = cv2.VideoCapture(VIDEO_PATH)

    # Output video properties
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if SAVE_OUTPUT:
        fps = int(video.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(OUT_PATH, fourcc, fps, (frame_width, frame_height))

    MOT_DETECTOR = Sort()

    for i in range(FRAME_COUNT):
        success, frame = video.read()

        # Continue until desired frame rate.
        if success:
            # Copy frame for display
            annotated_frame = copy.deepcopy(frame)
            # Convert frame to RGB for models
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Run human detection model
            humans_detected = model(frame, conf=CONF_LEVEL, classes=class_IDS)

            # Prepare detected persons with initial id's for MOT_DETECTOR
            # columns = ["x1", "y2", "x2", "y1", "conf", "class"]
            idx = [0, 1, 2, 3]
            pos_frame = humans_detected[0].boxes.data.numpy()[::, idx]

            # Update MOT_DETECTOR tracker object with respect to human detections
            track_bbs_ids = MOT_DETECTOR.update(pos_frame).astype(np.int32)

            # Containers to save detected workers
            worker_info = []  # (id, coord1, coord2)
            worker_images = []
            for person in track_bbs_ids:
                x1, y1, x2, y2, track_id = person

                # Crop and save persons from images
                # Expand the person according to expand constant.
                scale_expand = 0.2  # dogu
                len_expand_y = (y2 - y1) * scale_expand
                len_expand_x = (x2 - x1) * scale_expand
                y1_expanded = int(y1 - len_expand_y)
                y2_expanded = int(y2 + len_expand_y)
                x1_expanded = int(x1 - len_expand_x)
                x2_expanded = int(x2 + len_expand_x)

                # Exception handling for human bodies overflowing the boundries of the frame.
                if y1_expanded < 0:
                    y1_expanded = 0
                if y2_expanded > frame_height:
                    y2_expanded = frame_height - 1
                if x1_expanded < 0:
                    x1_expanded = 0
                if x2_expanded > frame_width:
                    x2_expanded = frame_width - 1

                # Cropping worker image
                worker_img = frame[y1_expanded:y2_expanded, x1_expanded:x2_expanded]
                # Save detected workers for equipment detection
                worker_images.append(worker_img)
                worker_id = person[4]
                worker_info.append((worker_id, (x1, y1), (x2, y2)))

            worker_objects = []  # Container to store worker objects
            # Run equipment detection for all workers
            for i in range(len(worker_images)):
                # Set coordinates
                worker_topLeft = worker_info[i][1]
                worker_bottomRight = worker_info[i][2]

                # Detect equipments
                worker_helmet = predict(worker_images[i], "helmet")  # status, conf
                worker_vest = predict(worker_images[i], "vest")  # status, conf

                equipments = {}
                equipments["helmet"] = worker_helmet
                equipments["vest"] = worker_vest

                worker_instance = Worker(
                    worker_topLeft, worker_bottomRight, worker_info[i][0], equipments
                )
                worker_objects.append(worker_instance)

            # Prepare display and show result
            annotated_frame = prepare_display(annotated_frame, worker_objects)
            cv2.imshow("Safety Equipment Detector", annotated_frame)

            if SAVE_OUTPUT == True:
                out.write(annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            continue

video.release()
cv2.destroyAllWindows()
