class Worker():
    
    """
    Attributes:
    -----------
    topLeftCoordinates : tuple (x,y)
        Coordinates for the top-left corner of the bounding box around the worker.
    bottomRightCoordinates : tuple (x,y)
        Coordinates for the bottom-right corner of the bounding box around the worker.
    workerID : str
        A unique identifier for the worker.
    equipments : dict ({equipment:(status, probability)})
        A dictionary containing detected equipments where keys are equipment names (e.g., "helmet", "vest") 
        and values are a tuple of (status, probability). Status is either 0 (not detected) or 1 (detected),
        and probability represents the model's confidence in the detection.

    Methods:
    --------
    is_equipment_detected(equipment_name: str) -> bool:
        Checks if a given equipment is detected for the worker.
    get_equipment_probability(equipment_name: str) -> float:
        Retrieves the probability/confidence of detection for a given equipment.
    get_coordinates() -> tuple:
        Retrieves the bounding box coordinates of the worker.
    """
    
    def __init__(self, topLeftCoordinates, bottomRightCoordinates, workerID, equipments):
        self.topLeftCoordinates = topLeftCoordinates
        self.bottomRightCoordinates = bottomRightCoordinates
        self.workerID = workerID
        self.equipments = equipments
    
    #temp
    def __repr__(self):
        id = "\nWorker: "+self.workerID
        coord = "\nCoordinates: "+ self.topLeftCoordinates + ", " + self.bottomRightCoordinates
        return "--------------------------------"+id+coord+"--------------------------------"
    
    def getCoordinates(self):
        return (self.topLefCoordinates, self.bottomRightCoordinates)
    
    def getWorkerID(self):
        return self.workerID
    
    def getEquipments(self):
        return self.equipments
    
    
    
        
        
        