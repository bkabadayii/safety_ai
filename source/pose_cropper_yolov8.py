import numpy as np
import cv2
from ultralytics import YOLO
import time
import copy

VIDEO_PATH = "../data/akcansa_share/video0.mp4"
# VIDEO_PATH = "../data/train6_nohelmet.jpg"
FRAME_COUNT = 1000
FRAME_RATE = 10

if __name__ == "__main__":
    model = YOLO("yolov8n-pose.pt")
    video = cv2.VideoCapture(VIDEO_PATH)

    for i in range(FRAME_COUNT):
        success, frame = video.read()

        if success:
            results = model(frame)

            # Try to update body parts
            for r in results:
                keypoints = r.keypoints.xy
                boxes = r.boxes.xyxy

            # annotated_frame = results[0].plot()
            annotated_frame = copy.deepcopy(frame)
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                current_keypoints = keypoints[i]

                # Crop head part
                top_y = int(y1)

                left_shoulder_Y = int(current_keypoints[5][1])
                right_shoulder_Y = int(current_keypoints[6][1])
                min_shoulder_Y = max(left_shoulder_Y, right_shoulder_Y)

                scale_head = 1.5
                head_height_new = int(scale_head * (top_y - min_shoulder_Y))

                # Crop body
                max_shoulder_Y = min(left_shoulder_Y, right_shoulder_Y)
                left_hip_Y = int(current_keypoints[11][1])
                right_hip_Y = int(current_keypoints[12][1])
                min_hip_Y = max(left_hip_Y, right_hip_Y)

                scale_body_coord = 0.2
                body_height = max_shoulder_Y - min_hip_Y
                body_low_Y, body_up_Y = int(
                    min_hip_Y - (scale_body_coord * body_height)
                ), int(max_shoulder_Y + (scale_body_coord * body_height))

                # Crop Hands
                left_wrist = current_keypoints[9]
                right_wrist = current_keypoints[10]

                scale_hand = 0.25
                hand_radius = (x2 - x1) * scale_hand

                right_hand_x1, right_hand_x2 = int(right_wrist[0] - hand_radius), int(
                    right_wrist[0] + hand_radius
                )
                right_hand_y1, right_hand_y2 = int(right_wrist[1] - hand_radius), int(
                    right_wrist[1] + hand_radius
                )

                left_hand_x1, left_hand_x2 = int(left_wrist[0] - hand_radius), int(
                    left_wrist[0] + hand_radius
                )
                left_hand_y1, left_hand_y2 = int(left_wrist[1] - hand_radius), int(
                    left_wrist[1] + hand_radius
                )

                # Draw rectangles

                # head
                cv2.rectangle(
                    annotated_frame,
                    (x1, (min_shoulder_Y + head_height_new)),
                    (x2, min_shoulder_Y),
                    color=(0, 255, 0),
                    thickness=2,
                )

                # body
                cv2.rectangle(
                    annotated_frame,
                    (x1, body_up_Y),
                    (x2, body_low_Y),
                    color=(255, 0, 0),
                    thickness=2,
                )
                """
                # hands
                cv2.rectangle(
                    annotated_frame,
                    (left_hand_x1, left_hand_y2),
                    (left_hand_x2, left_hand_y1),
                    color=(120, 0, 120),
                    thickness=2,
                )

                cv2.rectangle(
                    annotated_frame,
                    (right_hand_x1, right_hand_y2),
                    (right_hand_x2, right_hand_y1),
                    color=(120, 0, 120),
                    thickness=2,
                )
                """

            cv2.imshow("Body Detector", annotated_frame)
            cv2.imwrite("../data/akcansa_share/test.jpg", annotated_frame)
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

video.release()
cv2.destroyAllWindows()
