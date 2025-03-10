class person:
    
    filename: str
    detection_confidence: float
    gender_info: dict
    estimated_age: int
    race_info: dict    
    
    def __init__(self, filename, analysis):
        self.filename = filename
        self.detection_confidence = float(analysis["face_confidence"])
        detected_gender = analysis["dominant_gender"]
        self.gender_info = {
            "gender": detected_gender,
            "confidence": float(analysis["gender"][detected_gender])
        }
        self.estimated_age = analysis["age"]
        detected_races = {}
        for r, c in analysis["race"].items():
            if c > 20:                 # only include confidence levels over 20% for races
                detected_races[r] = float(c)
        self.race_info = {
            "dominant_race": analysis["dominant_race"],
            "race_confidences": detected_races
        }
        
    def getPersonInfo(self):
        personInfo = {
            "detection_confidence": self.detection_confidence,
            "gender_info": self.gender_info,
            "estimated_age": self.estimated_age,
            "race_info": self.race_info
        }
        return personInfo