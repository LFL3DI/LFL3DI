# LFL3DI

#  How to Custom Train the YOLOv8s model?

**Load the model.**

model = YOLO('yolov8s.pt')

**Training the custom model**

path = provide absolute path to the data.yaml file. This file is present in the Lidar-6 folder. This folder contains the custom taining, test and validation data

results = model.train(
   data= path,
   epochs=5,
   name='yolov8s_custom')


Results saved to runs\detect\yolov8s_custom

**Running the custom model**

path = provide absolute path to the best.pt present in the yolov8s_custom/weights/best.pt

model = YOLO(path)




