import cv2
import numpy as np
import copy
from PIL import Image, ImageDraw, ImageFont

# Class id 0: Helmet
# Class id 1: Vest

# Index 0 if equipment
# Index 1 if not equipment equipment
# Index 2 if not sure

COLORS = [(0, 142, 94), (161, 38, 45), (232, 191, 40)]
WORKER_FRAME_COLOR = (200, 0, 0)
WORKER_FRAME_THICKNESS = 2

EQUIPMENTS_COLOR = (200, 200, 0)
EQUIPMENTS_THICKNESS = 2
EQUIPMENTS_WIDTH = 100
EQUIPMENTS_HEIGHT = 100

TEXT_FONT = cv2.FONT_HERSHEY_PLAIN
TEXT_SIZE = 0.7
TEXT_THICKNESS = 2

# Fonts for different sizes
TITLE_FONT_BIG = ImageFont.truetype("../data/display_images/Barlow-Medium.ttf", 36)
TITLE_FONT_MEDIUM = ImageFont.truetype("../data/display_images/Barlow-Medium.ttf", 30)
TITLE_FONT_SMALL = ImageFont.truetype("../data/display_images/Barlow-Medium.ttf", 24)
BODY_FONT_BIG = ImageFont.truetype("../data/display_images/Barlow-Medium.ttf", 20)
BODY_FONT_MEDIUM = ImageFont.truetype("../data/display_images/Barlow-Medium.ttf", 16)
BODY_FONT_SMALL = ImageFont.truetype("../data/display_images/Barlow-Medium.ttf", 12)

# Display images
IMG = Image.open("../data/display_images/border1.png")
HELM_ICON = Image.open("../data/display_images/helm.png")
HELM_ICON = HELM_ICON.resize((80, 80))
VEST_ICON = Image.open("../data/display_images/vest.png")
VEST_ICON = VEST_ICON.resize((80, 80))
YELLOW_WARNING = Image.open("../data/display_images/yellow_warning.png")
RED_WARNING = Image.open("../data/display_images/red_warning.png")


# Draws transparent box inside an input image
def transparent_box(image, x1, y1, x2, y2, color=(0, 200, 0), alpha=0.4):
    overlay = copy.deepcopy(image)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)  # A filled rectangle

    # Following line overlays transparent rectangle over the image
    image_new = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return image_new


def decide_font(ratio):
    """
    @return: title_font, body_font
    """
    if ratio < 0.1:
        return TITLE_FONT_SMALL, BODY_FONT_SMALL
    elif ratio < 0.15:
        return TITLE_FONT_MEDIUM, BODY_FONT_MEDIUM

    return TITLE_FONT_BIG, BODY_FONT_BIG


def decide_warning_images_size(ratio):
    if ratio < 0.15:
        return 40
    elif ratio < 0.2:
        return 60

    return 80


def prepare_display(frame, workers):
    display_frame = copy.deepcopy(frame)
    frame_width = int(frame.shape[0])

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
        information_panel_height = worker_width * 9 // 16

        display_frame = cv2.rectangle(
            display_frame,
            (x1, y1),
            (x2, y2),
            (45, 38, 161),
            WORKER_FRAME_THICKNESS,
        )

        # Set equipments frame coordinates to be on the right
        equipments_x1, equipments_x2 = x1, x1 + EQUIPMENTS_WIDTH
        equipments_y1, equipments_y2 = y2, y1
        # If equipments frame overflows, put it on the left
        if equipments_x2 > frame_width:
            equipments_x1 = x1 - EQUIPMENTS_WIDTH
            equipments_x2 = x2

        # If it still overflows, set its x1 coordinates to be 0 (left most).
        if equipments_x1 < 0:
            equipments_x1 = 0

        # Put equipments frame
        display_frame = transparent_box(
            display_frame,
            equipments_x1,
            equipments_y1 - information_panel_height,
            x2,
            y2,
            color=(90, 90, 90),
            alpha=0.6,
        )

    temp_display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
    temp_display_frame = Image.fromarray(temp_display_frame)

    for worker in workers:
        x1, y1 = worker.topLeftCoordinates
        x2, y2 = worker.bottomRightCoordinates
        worker_width = x2 - x1
        worker_id = worker.workerID
        has_helmet, helmet_prob = worker.equipments["helmet"]
        has_vest, vest_prob = worker.equipments["vest"]

        helmet_color = COLORS[0] if has_helmet else COLORS[1]
        vest_color = COLORS[0] if has_vest else COLORS[1]

        # 0: No warning, 1: Yellow warning, 2: Red warning
        warning_level = 0
        if not has_helmet:
            warning_level += 1
        if not has_vest:
            warning_level += 1

        img = copy.deepcopy(IMG)
        information_panel_height = worker_width * 9 // 16
        img = img.resize((worker_width, information_panel_height))
        I1 = ImageDraw.Draw(img)
        height, width, _channel = np.array(img).shape
        helmet_icon = HELM_ICON.resize((int(0.1 * width), int(0.1 * width)))
        vest_icon = VEST_ICON.resize((int(0.1 * width), int(0.1 * width)))
        img.paste(helmet_icon, (int(width * 0.04), int(height * 0.35)))
        img.paste(vest_icon, (int(width * 0.04), int(height * 0.65)))
        coordIdX, coordIdY = width // 2, height * 0.1

        coordHelmX, coordHelmY = width * 0.15, height * 0.3
        coordHelmProbX, coordHelmProbY = width * 0.15, height * 0.45

        coordVestX, coordVestY = width * 0.15, height * 0.65
        coordVestProbX, coordVestProbY = width * 0.15, height * 0.8

        worker_frame_ratio = worker_width / frame_width
        title_font, body_font = decide_font(worker_frame_ratio)
        warnings_size = decide_warning_images_size(worker_frame_ratio)

        # Set texts
        I1.text(
            (coordIdX, coordIdY),
            text=f"Worker ID: {worker_id}",
            anchor="mt",
            font=title_font,
            fill=(14, 19, 10),
        )
        I1.text(
            (coordHelmX, coordHelmY),
            text="Helmet Stat:",
            font=body_font,
            fill=(14, 19, 10),
        )
        I1.text(
            (coordHelmX + 120, coordHelmY),
            text=f"{has_helmet}",
            font=body_font,
            fill=helmet_color,
        )
        I1.text(
            (coordHelmProbX, coordHelmProbY),
            text="Helmet Prob:",
            font=body_font,
            fill=(14, 19, 10),
        )
        I1.text(
            (coordHelmProbX + 120, coordHelmProbY),
            text=f"{np.round(helmet_prob, 2)}",
            font=body_font,
            fill=(14, 19, 19),
        )
        I1.text(
            (coordVestX, coordVestY),
            text="Vest Stat:",
            font=body_font,
            fill=(14, 19, 10),
        )
        I1.text(
            (coordVestX + 100, coordVestY),
            text=f"{has_vest}",
            font=body_font,
            fill=vest_color,
        )
        I1.text(
            (coordVestProbX, coordVestProbY),
            text="Vest Prob:",
            font=body_font,
            fill=(14, 19, 10),
        )
        I1.text(
            (coordVestProbX + 100, coordVestProbY),
            text=f"{np.round(vest_prob, 2)}",
            font=body_font,
            fill=(14, 19, 10),
        )

        temp_display_frame.paste(img, ((x1, y2 - information_panel_height)), img)

        if warning_level == 1:
            temp_display_frame.paste(
                YELLOW_WARNING.resize((warnings_size, warnings_size)),
                ((x1, y1)),
                YELLOW_WARNING.resize((warnings_size, warnings_size)),
            )
        elif warning_level == 2:
            temp_display_frame.paste(
                RED_WARNING.resize((warnings_size, warnings_size)),
                ((x1, y1)),
                RED_WARNING.resize((warnings_size, warnings_size)),
            )

    display_frame = np.array(temp_display_frame)
    return cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
