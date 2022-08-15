import torch
from torch import nn
from torchvision import models
from transformers import ViTFeatureExtractor, ViTModel, DeiTForImageClassificationWithTeacher
from torch.utils.data import DataLoader
from torchvision import transforms as T

import os
from PIL import Image
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def initialize_backbone(model_name, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0
    modules = None
    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(weights=use_pretrained)
        modules = list(model_ft.children())[:-1]
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(weights=use_pretrained)
        modules = list(model_ft.children())[:-1]
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(weights=use_pretrained)
        modules = list(model_ft.children())[:-1]
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(weights=use_pretrained)
        modules = list(model_ft.children())[:-1]
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(weights=use_pretrained)
        modules = list(model_ft.children())[:-1]
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(weights=use_pretrained)
        modules = list(model_ft.children())[:-1]
        input_size = 299

    elif model_name == "deit":
        deit = DeiTForImageClassificationWithTeacher.from_pretrained('facebook/deit-base-distilled-patch16-224')
        modules = list(deit.children())[:-2]
        input_size = None
    else:
        print("Invalid model name, exiting...")
        exit()

    return modules, input_size


class CustomImageDataset(Dataset):
    def __init__(self, embedding,y_train, train_path , transform=None, target_transform=None):
        self.img_labels = y_train
        self.img_dir = train_path
        self.img_embedding = embedding
        self.transform = transform
        # self.target_transform = target_transform
        
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_dir[idx]
        embedding = torch.tensor(self.img_embedding[idx], dtype=torch.float)
        # image = cv2.imread(img_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.open(img_path).convert('RGB')
        label = self.img_labels[idx]
        label = torch.tensor(int(label))
        if self.transform:
            image = self.transform(image)
        return image, embedding, label

class Network(nn.Module):
    def __init__(self, backbone):
      super(Network, self).__init__()
      # self.backbone = ViTModel.from_pretrained("google/vit-base-patch32-224-in21k")
      # modules = list(vit_model.children())[:-1]
      
      # model = SwinForImageClassification.from_pretrained(
      #          "microsoft/swin-tiny-patch4-window7-224")
      # self.backbone = nn.Sequential(*list(model.children())[:-1])

      #self.backbone = nn.Sequential(*modules)
      #self.backbone.out_channels = 768
      
      deit = DeiTForImageClassificationWithTeacher.from_pretrained('facebook/deit-base-distilled-patch16-224')
      modules = list(deit.children())[:-2]
      # modules = initialize_backbone(backbone)
      self.backbone = nn.Sequential(*modules)

      self.fc = nn.Linear(152064, 256)
      self.relu = nn.ReLU()
      self.dropout = nn.Dropout(0.1)
      self.fc2 = nn.Linear(1434, 256)
      self.fc3 = nn.Linear(512, 256)
      self.fc4 = nn.Linear(256, 5)
    
    def forward(self, x, y):
      x = self.backbone(x)
      print(x.last_hidden_state.shape)
      x = torch.flatten(x.last_hidden_state, 1)
      print(x.shape)
      x = self.fc(x)
      print(x.shape)
      x = self.relu(x)
      x = self.dropout(x)
      
      y = self.fc2(y)
      y = self.relu(y)
      y = self.dropout(y)
      
      z = torch.cat([x,y], axis=1)
      z = self.fc3(z)
      z = self.relu(z)
      z = self.fc4(z)
  
      return z

