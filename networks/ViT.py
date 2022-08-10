import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
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

class CustomImageDataset(Dataset):
    def __init__(self, train_path,y_train, embedding, transform=None, target_transform=None):
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
    def __init__(self):
      super(Network, self).__init__()
      # resnet_net = resnet50(weights=ResNet50_Weights.DEFAULT)
      # modules = list(resnet_net.children())[:-1]
      # self.backbone = ViTModel.from_pretrained("google/vit-base-patch32-224-in21k")
      # modules = list(vit_model.children())[:-1]
      
      # model = SwinForImageClassification.from_pretrained(
      #          "microsoft/swin-tiny-patch4-window7-224")
      # self.backbone = nn.Sequential(*list(model.children())[:-1])

      #self.backbone = nn.Sequential(*modules)
      #self.backbone.out_channels = 768
      
      deit = DeiTForImageClassificationWithTeacher.from_pretrained('facebook/deit-base-distilled-patch16-224')
      modules = list(deit.children())[:-2]
      self.backbone = nn.Sequential(*modules)

      self.fc = nn.Linear(152064, 256)
      self.relu = nn.ReLU()
      self.dropout = nn.Dropout(0.1)
      self.fc2 = nn.Linear(1434, 256)
      self.fc3 = nn.Linear(512, 256)
      self.fc4 = nn.Linear(256, 5)
    
    def forward(self, x, y):
      x = self.backbone(x)
      x = torch.flatten(x.last_hidden_state, 1)
      x = self.fc(x)
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

if __name__ == "__main__":

        X_train, y_train,train_path = load_data('data/FaceShape Dataset/training_set')
        X_test, y_test, test_path = load_data('data/FaceShape Dataset/testing_set')

        train_transforms = T.Compose([
            T.Resize((224,224)),
            #T.RandomResizedCrop((224,224)  ),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
                                            
        ])

        test_transforms = T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
                                            
        ])

        train = CustomImageDataset(train_path, y_train, X_train, transform=train_transforms) 
        test = CustomImageDataset(test_path, y_test, X_test, transform=test_transforms) 

        train_dataloader = DataLoader(train, batch_size=16, shuffle=True)
        test_dataloader = DataLoader(test, batch_size=16, shuffle=True)