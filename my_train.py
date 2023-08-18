import torch
from ultralytics import YOLO


if __name__ == '__main__':
    # Load a model
    model = YOLO("yolov8n.yaml")  # build a new model from scratch
    # model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    # Use the model
    model.train(data="coco128.yaml", device = 0,epochs=3,batch=1,imgsz=640,workers=0)  # train the model
    print(model)
    metrics = model.val()  # evaluate model performance on the validation set
    results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    path = model.export(format="onnx")  # export the model to ONNX format