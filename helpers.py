import torch
import pandas as pd 
import os 
from torchvision.transforms import v2
from PIL import Image
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
import sklearn.metrics as metrics

class networkTraining():
    def __init__(self, model, optimizer, criterion):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    #Adapted fromkuzu_main.py, hw1 of COMP9444
    def train(self, train_loader, epoch):
        self.model.train()
        correct = 0
        loss_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, target)
            loss.backward()
            self.optimizer.step()
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            loss_total += loss.item()
            
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

        print(f"Train Epoch: {epoch}\t Loss: {loss_total / len(train_loader.dataset):.6f} \t Accuracy: {100. * correct / len(train_loader.dataset):.0f}%")
    #Adapted fromkuzu_main.py, hw1 of COMP9444
    def test(self, test_loader):
        self.model.eval()
        test_loss = 0
        correct = 0
        #Number of categories in the dataset
        cats = test_loader.dataset.classes
        confusion = np.zeros((len(cats), len(cats)))
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                test_loss += self.criterion(outputs, target).item()
                pred = outputs.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                confusion += metrics.confusion_matrix(
                    target.cpu(), pred.cpu(), labels=[i for i in range(len(cats))])
            
            np.set_printoptions(precision=4, suppress=True)
            print(confusion)
        
        print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n")
        
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
            
        