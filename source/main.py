import numpy as np
import cv2
from ultralytics import YOLO
import copy

# object detector model
from object_detector import predict

#visualization
from visualize import transparent_box

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

#-------------------------------------------------------
### Configurations
#Verbose during prediction
verbose = False
# Scaling percentage of original frame
scale_percent = 100
tracker_threshold = 0.8
# Threshold of centers ( old\new)
thr_centers = 20
#Number of max frames to consider a object lost 
frame_max = 24
# Number of max tracked centers stored 
patience = 100
# ROI area color transparency
alpha = 0.1 #unused
#-------------------------------------------------------
# Reading video with cv2
video = cv2.VideoCapture(VIDEO_PATH)

# Objects to detect Yolo
class_IDS = [0] # default id for person is 0
# Auxiliary variables
centers_old = {}
obj_id = 0
end = []
frames_list = []
count_p = 0
lastKey = ''
print(f'[INFO] - Verbose during Prediction: {verbose}')
# -------------------------


#temp funcs
def detectWorkers():
    return 




def filter_tracks(centers, patience):
    """Function to filter track history"""
    filter_dict = {}
    for k, i in centers.items():
        d_frames = i.items()
        filter_dict[k] = dict(list(d_frames)[-patience:])

    return filter_dict


def update_tracking(centers_old,obj_center, thr_centers, lastKey, frame, frame_max):
    """Function to update track of objects"""
    is_new = 0
    lastpos = [(k, list(center.keys())[-1], list(center.values())[-1]) for k, center in centers_old.items()]
    lastpos = [(i[0], i[2]) for i in lastpos if abs(i[1] - frame) <= frame_max]
    # Calculating distance from existing centers points
    previous_pos = [(k,obj_center) for k,centers in lastpos if (np.linalg.norm(np.array(centers) - np.array(obj_center)) < thr_centers)]
    # if distance less than a threshold, it will update its positions
    if previous_pos:
        id_obj = previous_pos[0][0]
        centers_old[id_obj][frame] = obj_center
    
    # Else a new ID will be set to the given object
    else:
        if lastKey:
            last = lastKey.split('D')[1]
            id_obj = 'ID' + str(int(last)+1)
        else:
            id_obj = 'ID0'
            
        is_new = 1
        centers_old[id_obj] = {frame:obj_center}
        lastKey = list(centers_old.keys())[-1]

    
    return centers_old, id_obj, is_new, lastKey

if __name__ == "__main__":
    video = cv2.VideoCapture(VIDEO_PATH)

    # Output video properties
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if SAVE_OUTPUT:
        fps = int(video.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(OUT_PATH, fourcc, fps, (frame_width, frame_height))
    
    # VISUALZIE INPUTS:  Original frame, topleft(x1,y1) bottomright(x2, y2), workerid, classids, probs, TFx
        

    for i in range(FRAME_COUNT):
        success, frame = video.read()
        # Continue until desired frame rate.
        if CROP_AND_SAVE and (i % FRAME_RATE != 0):
            continue
        if success:
            
            detectWorkers()
            
            

            cv2.imshow("Safety Equipment Detector", annotated_frame)
            if(SAVE_OUTPUT==True):
                out.write(annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

video.release()
cv2.destroyAllWindows()
