from ultralytics import YOLO
model = YOLO(r"C:\Users\12731\Desktop\sd-webui-aki-v4.11.1-cu128\runs\detect\train-2\weights\best.pt")
odel.export(format="onnx")