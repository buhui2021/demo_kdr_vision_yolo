from ultralytics import YOLO
model = YOLO("yolo26n.yaml")
model.train(data=r"C:\Users\12731\Downloads\My First Project.yolo26\data.yaml", epochs=200, imgsz=640)