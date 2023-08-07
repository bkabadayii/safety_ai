from ultralytics import YOLO

safety_model = YOLO("../object_detector_model_training/models/C/weights/C.pt")
threshold = 0.40

def predict(img, targetClassName, threshold=threshold):
    
    results = safety_model(img)
    
    classNames = ["helmet", "vest"]

    for res in results:
        
        classIDs = res.boxes.cls
        probabilities = res.boxes.conf

        for classID, prob in zip(classIDs, probabilities):
            
            classID = int(classID.item())
            
            if(classNames[classID] == targetClassName):      
                prob = prob.item()
                #print(f"DETECTED --- Class: {classNames[int(classID)]}, Confidence: {prob}")
                if(prob>threshold): return True
    
    return False