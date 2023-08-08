import numpy as np
import cv2
from ultralytics import YOLO
import copy

import pandas as pd

# object detector model
from object_detector import predict

# Worker class
from worker import Worker

#display
from displays import prepare_display
#from PIL import Image as im

DEBUG_MODE = False

# Output video
#SAVE_OUTPUT = False
#OUT_PATH = f"../data/debug/result.mp4"

# 
VIDEO_NAME = "betonsa_3"
VIDEO_PATH = f"../data/video_data/{VIDEO_NAME}.mp4"

FRAME_COUNT = 1000

#-------------------------------------------------------
### Configurations
# Scaling percentage of original frame
CONF_LEVEL = 0.4
# Threshold of centers ( old\new)
THR_CENTERS = 200
#Number of max frames to consider a object lost 
FRAME_MAX = 24
# Number of max tracked centers stored 
PATIENCE = 100
# ROI area color transparency
ALPHA = 0.1 #unused
#-------------------------------------------------------
# Reading video with cv2
video = cv2.VideoCapture(VIDEO_PATH)

# Objects to detect Yolo
class_IDS = [0] # default id for person is 0

# Auxiliary variables
centers_old = {}
obj_id = 0
count_p = 0
last_key = ''
#-------------------------------------------------------

#temp funcs
def detectWorkers():
    return 

def filter_tracks(centers, PATIENCE):
    """Function to filter track history"""
    filter_dict = {}
    for k, i in centers.items():
        d_frames = i.items()
        filter_dict[k] = dict(list(d_frames)[-PATIENCE:])

    return filter_dict


def update_tracking(centers_old,obj_center, THR_CENTERS, last_key, frame, FRAME_MAX):
    """Function to update track of objects"""
    is_new = 0
    lastpos = [(k, list(center.keys())[-1], list(center.values())[-1]) for k, center in centers_old.items()]
    lastpos = [(i[0], i[2]) for i in lastpos if abs(i[1] - frame) <= FRAME_MAX]
    # Calculating distance from existing centers points
    previous_pos = [(k,obj_center) for k,centers in lastpos if (np.linalg.norm(np.array(centers) - np.array(obj_center)) < THR_CENTERS)]
    # if distance less than a threshold, it will update its positions
    if previous_pos:
        id_obj = previous_pos[0][0]
        centers_old[id_obj][frame] = obj_center
    
    # Else a new ID will be set to the given object
    else:
        if last_key:
            last = last_key.split('D')[1]
            id_obj = 'ID' + str(int(last)+1)
        else:
            id_obj = 'ID0'
            
        is_new = 1
        centers_old[id_obj] = {frame:obj_center}
        last_key = list(centers_old.keys())[-1]

    
    return centers_old, id_obj, is_new, last_key


#loading a YOLO model 
model = YOLO('yolov8n.pt')

#geting names from classes
dict_classes = model.model.names

if __name__ == "__main__":
    video = cv2.VideoCapture(VIDEO_PATH)

    # Output video properties
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    for i in range(FRAME_COUNT):
        success, frame = video.read()
        
        # Continue until desired frame rate.
        if success:
            
            annotated_frame = copy.deepcopy(frame)
            y_hat = model.predict(frame, conf = CONF_LEVEL, classes = class_IDS)

            boxes   = y_hat[0].boxes.xyxy.cpu().numpy()
            conf    = y_hat[0].boxes.conf.cpu().numpy()
            classes = y_hat[0].boxes.cls.cpu().numpy()

            # Storing the above information in a dataframe
            positions_frame = pd.DataFrame(y_hat[0].cpu().numpy().boxes.boxes, columns = ['xmin', 'ymin', 'xmax', 'ymax', 'conf', 'class'])
            
            #Translating the numeric class labels to text
            labels = [dict_classes[i] for i in classes]
            
            worker_info = [] # (id, coord1, coord2)
            worker_Images = []
            
            # For each people, draw the bounding-box and add scaled and cropped images to list
            for ix, row in enumerate(positions_frame.iterrows()):
                # Getting the coordinates of each vehicle (row)
                x1, y2, x2, y1, confidence, category,  = row[1].astype('int')
                
                # Calculating the center of the bounding-box
                center_x, center_y = int(((x2+x1))/2), int((y1+ y2)/2)
                
                #Updating the tracking for each object
                centers_old, id_obj, is_new, last_key = update_tracking(centers_old, (center_x, center_y), THR_CENTERS, last_key, i, FRAME_MAX)
                
                #Updating people in roi
                count_p+=is_new

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
                workerImg = frame[y2_expanded:y1_expanded, x1_expanded:x2_expanded]

                # Drawing center and bounding-box in the given frame 
                cv2.rectangle(annotated_frame, (x1, y2), (x2, y1), (0,0,255), 2) # box
                """
                for center_x,center_y in centers_old[id_obj].values():
                    cv2.circle(annotated_frame, (center_x,center_y), 5,(0,0,255),-1) # center of box
                """
                
                # Drawing above the bounding-box the name of class recognized.
                """
                cv2.putText(img=annotated_frame, text=id_obj+':'+str(np.round(conf[ix],2)),
                            org= (x1,y2-10), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.8, color=(0, 0, 255),thickness=1)
                """
                

                worker_Images.append(workerImg)
                worker_info.append((last_key, (x1,y1), (x2,y2)))

            # for worker in workers_cropped -> predict yap -> worker objects listesine ekle
            worker_objects = []
            
            for i in range(len(worker_Images)):
                
                #coordinates
                worker_topLeft = worker_info[i][1]
                worker_bottomRight = worker_info[i][2]
                
                #equipments
                worker_helmet = predict(worker_Images[i], "helmet") # status, conf
                worker_vest = predict(worker_Images[i], "vest") # status, conf
                
                equipments = {}
                equipments["helmet"] = worker_helmet 
                equipments["vest"] = worker_vest
                
                worker_instance = Worker(worker_topLeft, worker_bottomRight, worker_info[i][0], equipments)
                worker_objects.append(worker_instance)
                
            
            # display
            annotated_frame = prepare_display(annotated_frame, worker_objects)
            
            #drawing the number of people
            """
            cv2.putText(img=annotated_frame, text=f'Counts People in ROI: {count_p}', 
                        org= (30,40), fontFace=cv2.FONT_HERSHEY_TRIPLEX, 
                        fontScale=1.5, color=(255, 0, 0), thickness=1)
            """

            # Filtering tracks history
            centers_old = filter_tracks(centers_old, PATIENCE)
            
            cv2.imshow("Safety Equipment Detector", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            continue

video.release()
cv2.destroyAllWindows()