import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import  torchvision.datasets
import torchvision.models as models
from model import *


img_path = "../Test/" # location to the image path.
image_dataset = torchvision.datasets.ImageFolder(img_path, transform=torchvision.transforms.ToTensor())


model_load = CNN()
model_data = torch.load('../model/CNN') # location to the save CNN using the checkpoint
model_load.load_state_dict(model_data)

mapping = {
  0:"A",
  1: "B",
  2: "C",
  3: "D",
  4: "E",
  5: "F",
  6: "G",
  7: "H",
  8: "I",
  9: "J",
  10: "K",
  11: "L",
  12: "M",
  13: "N",
  14: "O",
  15: "P",
  16: "Q",
  17: "R",
  18: "S",
  19: "T",
  20: "U",
  21: "V",
  22: "W",
  23: "X",
  24: "V",
  25: "Z",
  26: "del",
  27: "nothing",
  28: "space"
}

def get_accuracy(model, data):
    
    loader = torch.utils.data.DataLoader(data, batch_size=1)
    model.eval()
    for img, label in loader:

      # annotate model for evaluation
      output = model(img) # We don't need to run torch.softmax
      pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
      pred = pred.detach().numpy()
      output_pred = pred[0][0]
      print("Model Output: {} || Mapping Alphabet: {}".format(output_pred,mapping[output_pred]))
    return None

print(get_accuracy(model_load, image_dataset))
