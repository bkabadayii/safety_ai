from ultralytics import YOLO

modelC = YOLO("../model_training/models/C/weights/C.pt")
modelV = YOLO("../model_training/models/V/weights/V.pt")

threshold = 0.40

# Both of the 'run' functoins may also return the whole list of probabilities, and the class ids for images that contain multiple person
# currently, it assumes that img contains maximum 1 person in the frame. (our current solution)
# You can also use predictLegacy() for the images that contain multiple person for now.

def predict(img, targetClassName, threshold=threshold):
    
    if(targetClassName=="vest"):
        results = runVestModel(img)
    else:
        results = runHelmetModel(img)
    
    return results # (Boolean, Float)

def runVestModel(img):
    
    results = modelV(img)
    
    for res in results:
        if(len(res.boxes.cls)>0):
            vest_status = res.boxes.cls[0].item() == 1
            status_prob = res.boxes.conf[0]
        else:
            vest_status = False
            status_prob = 0.0
    
        return (vest_status, status_prob)

    return (False, 0.0)

def runHelmetModel(img):
    
    
    results = modelC(img)
    
    for res in results:
        if(len(res.boxes.cls)>0):
            helmet_status = res.boxes.cls[0].item() == 0 # 0 for 'helmet' and 1 for 'vest' in modelC.
            status_prob = res.boxes.conf[0]
        else:
            helmet_status = False
            status_prob = 0.0
        
        return (helmet_status, status_prob)
    
    return (False, 0.0)

# UNUSED
def predictLegacy(img, targetClassName, threshold=threshold):
    
    results = modelC(img)
    
    classNames = ["helmet", "vest"]
    for res in results:
        
        classIDs = res.boxes.cls
        probabilities = res.boxes.conf

        for classID, prob in zip(classIDs, probabilities):
            
            classID = int(classID.item())
            
            if(classNames[classID] == targetClassName):                      
                #Debug #print(f"DETECTED --- Class: {classNames[int(classID)]}, Confidence: {prob}")
                
                prob = prob.item()  
                if(prob > threshold):
                    return (True, prob)
                else:
                    return (False, prob)
    
    return (False, 0)