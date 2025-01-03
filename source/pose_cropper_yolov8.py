import numpy as np
import cv2
from ultralytics import YOLO
import copy

# object detector model
from object_detector import predict

DEBUG_MODE = False

# Output video
SAVE_OUTPUT = True
OUT_PATH = f"../data/debug/result.mp4"


VIDEO_NAME = "betonsa_3"
VIDEO_PATH = f"../data/video_data/{VIDEO_NAME}.mp4"

CROP_AND_SAVE = False  # True if you want to crop and save body parts
SAVE_PATH = f"../debugging/cropped_parts/{VIDEO_NAME}"

FRAME_COUNT = 1000
FRAME_RATE = 20

model = YOLO("yolov8n-pose.pt")

# Draws transparent box inside an input image
def transparent_box(image, x, y, w, h, color=(0, 200, 0), alpha=0.4):
    overlay = copy.deepcopy(image)
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)  # A filled rectangle

    # Following line overlays transparent rectangle over the image
    image_new = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return image_new


if __name__ == "__main__":
    video = cv2.VideoCapture(VIDEO_PATH)

    # Output video properties
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if SAVE_OUTPUT:
        fps = int(video.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(OUT_PATH, fourcc, fps, (frame_width, frame_height))

    for i in range(FRAME_COUNT):
        success, frame = video.read()
        # Continue until desired frame rate.
        if CROP_AND_SAVE and (i % FRAME_RATE != 0):
            continue
        if success:
            results = model(frame)

            # Try to update body parts
            for r in results:
                keypoints = r.keypoints.xy
                boxes = r.boxes.xyxy

            # annotated_frame = results[0].plot()
            annotated_frame = copy.deepcopy(frame)
            for j, box in enumerate(boxes):
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                current_keypoints = keypoints[j]

                # Crop head part
                top_y = int(y1)

                left_shoulder_Y = int(current_keypoints[5][1])
                right_shoulder_Y = int(current_keypoints[6][1])
                min_shoulder_Y = max(left_shoulder_Y, right_shoulder_Y)

                scale_head = 1.2
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

                """
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
                """

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
                if body_up_Y < 0:
                    body_up_Y = 0
                if body_low_Y < 0:
                    body_lowY = 0

                # Crop and save the image.
                person_cropped = frame[y1_expanded:y2_expanded, x1_expanded:x2_expanded]

                if CROP_AND_SAVE:
                    cv2.imwrite(f"{SAVE_PATH}/person_{i}_{j}.jpg", person_cropped)

                head_cropped = frame[y1_expanded:body_up_Y, x1_expanded:x2_expanded]
                body_cropped = frame[body_up_Y:body_low_Y, x1_expanded:x2_expanded]

                # Predict
                helmet_status, helmet_prob = predict(person_cropped, "helmet")
                vest_status, vest_prob = predict(person_cropped, "vest")

                # Set color
                helmet_color = (0, 200, 0) if helmet_status else (0, 0, 200)
                vest_color = (0, 200, 0) if vest_status else (0, 0, 200)

                # Display helmet status
                annotated_frame = transparent_box(
                    annotated_frame,
                    x1,
                    body_up_Y,
                    x2 - x1,
                    head_height_new - (body_up_Y - min_shoulder_Y),
                    color=helmet_color,
                )

                # Display vest status
                annotated_frame = transparent_box(
                    annotated_frame,
                    x1,
                    body_low_Y,
                    x2 - x1,
                    body_up_Y - body_low_Y,
                    color=vest_color,
                )
                
                # Display texts
                helmet_text = f"Helmet: {'Detected' if helmet_status else 'Not Detected'}, Prob: {helmet_prob:.2f}"
                cv2.putText(annotated_frame, helmet_text, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, helmet_color, 2)
                
                vest_text = f"Vest: {'Detected' if vest_status else 'Not Detected'}, Prob: {vest_prob:.2f}"
                cv2.putText(annotated_frame, vest_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, vest_color, 2)

                # Draw rectangles
                # head
                cv2.rectangle(
                    annotated_frame,
                    (x1, (min_shoulder_Y + head_height_new)),
                    (x2, body_up_Y),
                    color=helmet_color,
                    thickness=2,
                )

                # body
                cv2.rectangle(
                    annotated_frame,
                    (x1, body_up_Y),
                    (x2, body_low_Y),
                    color=vest_color,
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

                # Debug imwrites
                if DEBUG_MODE and j == 0:
                    cv2.imwrite("../data/debug/test.jpg", annotated_frame)
                    cv2.imwrite("../data/debug/head_cropped.jpg", head_cropped)
                    cv2.imwrite("../data/debug/body_cropped.jpg", body_cropped)

            cv2.imshow("Safety Equipment Detector", annotated_frame)
            if(SAVE_OUTPUT==True):
                out.write(annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

video.release()
cv2.destroyAllWindows()
