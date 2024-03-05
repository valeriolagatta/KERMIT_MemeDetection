#Definisco qui il modello di Object Detection
from imageai.Detection import ObjectDetection
import os

#Modello da utilizzare
path_drive_YOLO = "yolo.h5"


detector = ObjectDetection()
detector.setModelTypeAsYOLOv3() #Risulta essere veloce e mediamente accurato
detector.setModelPath(path_drive_YOLO)
detector.loadModel()

# Definisco la funzione di Object Recognition
def ObjectRecognizer(path_img):
    detections = detector.detectObjectsFromImage(input_image=path_img, output_image_path="dummy_file.png")

    tags = []
    for eachObject in detections:
        tags.append(eachObject["name"])

    return tags  # Ritorno i tags con gli elementi trovati
