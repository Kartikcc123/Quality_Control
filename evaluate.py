import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from model import FabricCNN

classes = ["good","hole","objects","oil_spot","thread_error"]

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64,64)),
    transforms.ToTensor()
])

test_dataset = datasets.ImageFolder("dataset/test", transform=transform)

test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

model = FabricCNN()
model.load_state_dict(torch.load("fabric_cnn_model.pth"))

model.eval()

y_true = []
y_pred = []

with torch.no_grad():

    for images, labels in test_loader:

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.numpy())

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes,
            yticklabels=classes)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()