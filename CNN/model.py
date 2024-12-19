import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.optim as optim

from torchvision.models import resnet50, ResNet50_Weights
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class ImageClassificationModel_FC(nn.Module):
    def __init__(self):
        super(ImageClassificationModel_FC, self).__init__()
        self.fc1 = nn.Linear(3 * 300 * 300, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128 ,128)
        self.fc4 = nn.Linear(128 ,128)
        self.fc5 = nn.Linear(128 ,128)
        self.fc6 = nn.Linear(128 ,128)
        
        self.dropout = nn.Dropout(0.5)
        
        self.fcm = nn.Linear(128, 3) 
    
    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        
        x = self.dropout(x)
        
        x = self.fcm(x)
        
        return x

class ImageClassificationModel_CNN(nn.Module):
    def __init__(self):
        super(ImageClassificationModel_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding = 0)
        self.dropout = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(64 * 37 * 37, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        
        x = x.view(-1, 64 * 37 * 37)
        
        x = self.dropout(x)
        
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

class building_block_method(nn.Module):
    def __init__(self, num_classes, fc_requires_grad=True):
        super(building_block_method, self).__init__()
        
        resnet_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        in_features = resnet_model.fc.in_features
        resnet_model.fc = nn.Identity()
        
        for param in resnet_model.parameters():
            param.requires_grad = False
        
        self.resnet = resnet_model
        self.fc1 = nn.Linear(in_features, 32)
        self.fc1.requires_grad = False
        self.fc2 = nn.Linear(32, 64)
        self.fc2.requires_grad = fc_requires_grad
        self.fc3 = nn.Linear(64, num_classes)
        self.fc3.requires_grad = fc_requires_grad
        
    def forward(self,x):
        x = self.resnet(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

class ImageDataLoader:
    def __init__(self):
        transform = transforms.Compose([
        transforms.Resize((300, 300)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        root_dir = 'ProjectImages'
        full_train_dataset = ImageFolder(root=root_dir, transform=transform)
        self.classes = full_train_dataset.classes
        
        total_size = len(full_train_dataset)
        train_size = int(total_size * 0.7)
        val_size = int(total_size * 0.1)
        test_size = total_size - train_size - val_size
        
        self.train_data, self.val_data, self.test_data = random_split(full_train_dataset, [train_size, val_size, test_size])
        
        self.train_loader = DataLoader(self.train_data, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
        self.val_loader = DataLoader(self.val_data, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
        self.test_loader = DataLoader(self.test_data, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
        
class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, classes, epochs=10, learning_rate=0.001):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.classes = classes
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        self.epochs = epochs

        self.train_losses = []
        self.val_losses = []
    
    def train(self, device):
        self.model.to(device)
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0
            with tqdm(total=len(self.train_loader), desc=f"Epoch {epoch + 1}/{self.epochs}", unit="batch", leave=False) as pbar:
                for features, labels in self.train_loader: 
                    features, labels = features.to(device), labels.to(device)
                    outputs = self.model(features)  
                    loss = self.criterion(outputs, labels)
                    train_loss += loss.item()

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step() 
                    
                    pbar.set_postfix({'train_loss': f'{loss.item():.4f}'})
                    pbar.update(1)

            train_loss /= len(self.train_loader)
            val_loss = self.validate(device)   
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)            
            
            print(f"Epoch {epoch + 1}/{self.epochs} - Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
    
    def validate(self, device):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, labels in self.val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
        
        val_loss /= len(self.val_loader)
        return val_loss

    def plot_loss(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.legend()
        plt.show()
    
    def evaluate(self, device):
        self.model.eval()
        test_loss = 0
        all_labels = []
        all_preds = []

        total = 0
        correct = 0
        
        correct_predictions = []
        incorrect_predictions = []
        
        with torch.no_grad():
            for features, labels in self.test_loader:
                features, labels = features.to(device), labels.to(device)         
                outputs = self.model(features.float())
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()
                
                output_prob = F.softmax(outputs, dim=1).cpu().numpy()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
            
                for f, l, p, o, prob in zip(features, labels, predicted, outputs, output_prob):
                    if l == p and len(correct_predictions) < 10:
                        correct_predictions.append((f, l, p, o, prob))
                    elif l != p and len(incorrect_predictions) < 10:
                        incorrect_predictions.append((f, l, p, o, prob))
                    if len(correct_predictions) >= 10 and len(incorrect_predictions) >= 10:
                        break
        
        print("\n----- Correct Predictions -----")
        for f, l, p, o, prob in correct_predictions:
            f_permute = f.permute(1, 2, 0).cpu().numpy()
            
            restored_image = (f_permute * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
            restored_image = np.clip(restored_image, 0, 1)
            
            plt.imshow(restored_image, cmap='gray')
            plt.title(f'Predicted: {self.classes[p]}, Actual: {self.classes[l]}')
            plt.show()            
            print(f"{o}->{prob}: {self.classes[l]} -> {self.classes[p]}")
        
        print("\n----- Incorrect Predictions -----")
        for f, l, p, o, prob in incorrect_predictions:
            f_permute = f.permute(1, 2, 0).cpu().numpy()
            
            restored_image = (f_permute * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
            restored_image = np.clip(restored_image, 0, 1)
            
            plt.imshow(restored_image, cmap='gray')
            plt.title(f'Predicted: {self.classes[p]}, Actual: {self.classes[l]}')
            plt.show()            
            print(f"{o}->{prob}: {self.classes[l]} -> {self.classes[p]}")
        
        test_loss /= len(self.test_loader)
        print(f"Accuracy = {correct/total: .4f}")

        cm = confusion_matrix(all_labels, all_preds)
        print(f"Test Loss: {test_loss:.4f}")
        print("Confusion Matrix:")
        print(cm)

if __name__ == '__main__':
    device = torch.device('mps') if torch.backends.mps.is_available() else "cpu"
    print(device)
    
    model1 = ImageClassificationModel_FC()
    model1.to(device)
    
    model2 = ImageClassificationModel_CNN()
    model2.to(device)
    
    resnet50(weights=ResNet50_Weights.DEFAULT)
    model3 = building_block_method(num_classes=3, fc_requires_grad=True)
    model3.to(device)
    
    dataloader = ImageDataLoader()

    trainer1 = ModelTrainer(model1, dataloader.train_loader, dataloader.val_loader, dataloader.test_loader, dataloader.classes, epochs=2)
    trainer1.train(device)
    trainer1.validate(device)
    trainer1.plot_loss()
    trainer1.evaluate(device)
    
    trainer2 = ModelTrainer(model2, dataloader.train_loader, dataloader.val_loader, dataloader.test_loader, dataloader.classes, epochs=2)
    trainer2.train(device)
    trainer2.validate(device)
    trainer2.plot_loss()
    trainer2.evaluate(device)
    
    trainer3 = ModelTrainer(model3, dataloader.train_loader, dataloader.val_loader, dataloader.test_loader, dataloader.classes, epochs=2)
    trainer3.train(device)
    trainer3.validate(device)
    trainer3.plot_loss()
    trainer3.evaluate(device)
