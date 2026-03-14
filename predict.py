import torch
import cv2
import numpy as np
from model import FabricCNN

classes = ["good","hole","objects","oil_spot","thread_error"]

model = FabricCNN()
model.load_state_dict(torch.load("fabric_cnn_model.pth"))
model.eval()

img = cv2.imread("test2.jpeg",0)
img = cv2.resize(img,(64,64))

img = img/255.0

img = torch.tensor(img).float().unsqueeze(0).unsqueeze(0)

output = model(img)

pred = torch.argmax(output)

print("Prediction:",classes[pred])