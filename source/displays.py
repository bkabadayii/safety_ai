import cv2
import numpy as np
import copy

# Class id 0: Helmet
# Class id 1: Vest

# Index 0 if equipment
# Index 1 if not equipment equipment
# Index 2 if not sure

COLORS = [(0, 200, 0), (0, 0, 200), (200, 200, 0)]
WORKER_FRAME_COLOR = (200, 0, 0)
WORKER_FRAME_THICKNESS = 2

EQUIPMENTS_COLOR = (200, 200, 0)
EQUIPMENTS_THICKNESS = 2
EQUIPMENTS_WIDTH = 100
EQUIPMENTS_HEIGHT = 100

TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX


# Draws transparent box inside an input image
def transparent_box(image, x1, y1, x2, y2, color=(0, 200, 0), alpha=0.4):
    overlay = copy.deepcopy(image)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)  # A filled rectangle

    # Following line overlays transparent rectangle over the image
    image_new = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return image_new


def prepare_display(frame, workers):
    display_frame = copy.deepcopy(frame)
    frame_width = int(frame.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(frame.get(cv2.CAP_PROP_FRAME_HEIGHT))
    for worker in workers:
        worker_id = worker.workerID
        has_helmet, helmet_prob = worker.equipments["helmet"]
        has_vest, vest_prob = worker.equipments["vest"]
        x1, y1 = worker.topLeftCoordinates
        x2, y2 = worker.bottomRightCoordinates
        worker_width = x2 - x1
        worker_height = y1 - y2

        helmet_color = COLORS[0] if has_helmet else COLORS[1]
        vest_color = COLORS[0] if has_vest else COLORS[1]
        equipments_color = 0
        if has_helmet and has_vest:
            equipments_color = COLORS[0]
        elif has_helmet or has_vest:
            equipments_color = COLORS[2]
        else:
            equipments_color = COLORS[1]

        display_frame = cv2.rectangle(
            display_frame,
            (x1, y1),
            (x2, y2),
            WORKER_FRAME_COLOR,
            WORKER_FRAME_THICKNESS,
        )

        # Set equipments frame coordinates to be on the right
        equipments_x1, equipments_x2 = x1, x1 + EQUIPMENTS_WIDTH
        equipments_y1, equipments_y2 = y1, y2
        # If equipments frame overflows, put it on the left
        if equipments_x2 > frame_width:
            equipments_x1 = x1 - EQUIPMENTS_WIDTH
            equipments_x2 = x2

        # If it still overflows, set its x1 coordinates to be 0 (left most).
        if equipments_x1 < 0:
            equipments_x1 = 0

        # Put worker id text
        cv2.putText(
            display_frame,
            f"Worker ID: {worker_id}",
            (equipments_x1 + 5, equipments_y1 - 5),
            TEXT_FONT,
            1,
            equipments_color,
            2,
        )

        # Put helmet status text
        cv2.putText(
            display_frame,
            f"Helmet: {has_helmet}, probability: {round(helmet_prob, 2)}",
            (equipments_x1 + 5, equipments_y1 - 15),
            TEXT_FONT,
            1,
            helmet_color,
            2,
        )

        # Put vest status text
        cv2.putText(
            display_frame,
            f"Vest: {has_vest}, probability: {round(vest_prob, 2)}",
            (equipments_x1 + 5, equipments_y1 - 25),
            TEXT_FONT,
            1,
            equipments_color,
            2,
        )
    return display_frame
