import numpy as np
import cv2
from keras.models import load_model
import copy

# Load face cascade
face_cascade = cv2.CascadeClassifier("../xml_files/haarcascade_frontalface_default.xml")

# Load the model
model = load_model("../final_model/keras_model.h5")

# Grab the labels from the labels.txt file. This will be used later.
labels = open("../final_model/labels.txt", "r").readlines()

# Camera
camera = cv2.VideoCapture(0)


while True:
    # Load the image and convert it to grayscale.
    ret, image = camera.read()

    if ret == False:
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Generate display image
    display = copy.deepcopy(image)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        print("Face Not Found!")

    cropped_faces = []
    cropped_faces_coords = []
    # Iterate over the detected faces

    for x, y, w, h in faces:
        # Crop the image (0.5 * length_face) cm above the face
        y_offset = int(0.5 * h)  # 0.5 * length_face above the face
        crop_img = image[y - y_offset : y + h, x : x + w]
        try:
            cropped_faces.append(crop_img)
            cropped_faces_coords.append((x, y, w, h))
        except:
            pass

    # DETECT WHETHER THE FACE HAS HELMET:

    # Array to store faces' classifications
    face_labels = []
    face_sureness = []
    try:
        for image in cropped_faces:
            # Resize the raw image into (224-height,224-width) pixels.
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
            # Make the image a numpy array and reshape it to the models input shape.
            image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
            # Normalize the image array
            image = (image / 127.5) - 1

            # Have the model predict what the current image is. Model.predict
            # returns an array of percentages. Example:[0.2,0.8] meaning its 20% sure
            # it is the first label and 80% sure its the second label.
            probabilities = model.predict(image)
            # Print what the highest value probability label
            label = labels[np.argmax(probabilities)]
            sureness = max(probabilities[0]) * 100

            print(f"Sureness level: {max(probabilities[0]) * 100} %")
            print(label)

            face_labels.append(int(label[0]))
            face_sureness.append(str(round(sureness, 2)))

    except:
        pass

    # Prepare the display image
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5

    for i, face_label in enumerate(face_labels):
        color = (0, 0, 0)
        text = "Unknown"
        if face_label == 0:
            color = (0, 255, 0)
            text = "Helmet"
        else:
            color = (0, 0, 255)
            text = "No Helmet"

        # Put rectangles:
        x, y, w, h = cropped_faces_coords[i]
        cv2.rectangle(display, (x, y), (x + w, y + h), color, thickness=thickness)

        # Put text:
        text_x = x + (w // 2) - 20
        text_y = y - 10

        # Label
        cv2.putText(display, text, (text_x, text_y), font, fontScale, color, thickness)

        # Sureness
        cv2.putText(
            display,
            face_sureness[i],
            (text_x, text_y - 20),
            font,
            fontScale,
            color,
            thickness,
        )

    cv2.imshow("Helmet Detector", display)

    if cv2.waitKey(1) == 27:
        break

camera.release()
cv2.destroyAllWindows()
