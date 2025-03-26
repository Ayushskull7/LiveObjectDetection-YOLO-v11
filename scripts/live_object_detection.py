import numpy as np
import cv2 as cv
from ultralytics import YOLO
import torch
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=int, default=0, help='Video source, default is 0 for webcam')
parser.add_argument('--save', type=str, default='output.mp4', help='Filename for saved video')
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

script_dir = os.path.dirname(os.path.abspath(__file__))  
model_path = os.path.join(script_dir, "..", "models", "yolo11x.pt")  
model = YOLO(model_path).to(device)

objects_path = os.path.join(script_dir, "objects.txt")
with open(objects_path, 'r') as file:
    object_names_to_detect = set(line.strip() for line in file)

cap = cv.VideoCapture(args.source)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

fourcc = cv.VideoWriter_fourcc(*'mp4v')
output_dir = os.path.join(script_dir, "..", "output")
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, args.save)
out = cv.VideoWriter(save_path, fourcc, 60.0, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    results = model(frame)
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            object_name = model.names[cls]
            if conf > 0.5 and object_name in object_names_to_detect:
                label = f"{object_name} {conf:.2f}"
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv.putText(frame, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv.imshow('frame', frame)
    out.write(frame)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
cv.destroyAllWindows()
