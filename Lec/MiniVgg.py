import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniVggBnBefore(nn.Module):
    def __init__(self):
        super(MiniVggBnBefore, self).__init__()
        
        # first: CONV => RELU => CONV => RELU => POOL set
        self.conv1_1 = nn.Conv2d(1, 32, 3, padding = 1)
        self.norm1_1 = nn.BatchNorm2d(32)
    
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding = 1)
        self.norm1_2 = nn.BatchNorm2d(32)
    
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout2d(0.25)
        
        # second: CONV => RELU => CONV => RELU => POOL set
        self.conv2_1 = nn.Conv2d(32, 64, 3, padding = 1)
        self.norm2_1 = nn.BatchNorm2d(64)
    
        self.conv2_2 = nn.Conv2d(64, 128, 3, padding = 1)
        self.norm2_2 = nn.BatchNorm2d(128)
    
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.dropout2 = nn.Dropout2d(0.25)
        
        # fully connected (single) to RELU
        
        self.fc1 = nn.Linear(128 * 7 * 7, 128)
        self.normfc_1 = nn.BatchNorm1d(128)    
        self.dropoutfc_1 = nn.Dropout1d(0.20)
        
        self.fc2 = nn.Linear(128, 10)
        
        
    def forward(self, x):          
        out = F.relu(self.norm1_1(self.conv1_1(x)))
        out = F.relu(self.norm1_2(self.conv1_2(out)))
        out = self.pool1(out)
        out = self.dropout1(out)
        
        out = F.relu(self.norm2_1(self.conv2_1(out)))
        out = F.relu(self.norm2_2(self.conv2_2(out)))
        out = self.pool2(out)
        out = self.dropout2(out)
        
        # flatten
        out = out.view(-1, 128 * 7 * 7)
        
        out = F.relu(self.normfc_1(self.fc1(out)))
        out = self.dropoutfc_1(out)
        
        out = self.fc2(out)
        
        # softmax classifier
        
        return out
    
    
class MiniVggBnAfter(nn.Module):
    def __init__(self):
        super(MiniVggBnAfter, self).__init__()
        
        # first: CONV => RELU => CONV => RELU => POOL set
        self.conv1_1 = nn.Conv2d(1, 32, 3, padding = 1)
        self.norm1_1 = nn.BatchNorm2d(32)
    
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding = 1)
        self.norm1_2 = nn.BatchNorm2d(32)
    
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout2d(0.25)
        
        # second: CONV => RELU => CONV => RELU => POOL set
        self.conv2_1 = nn.Conv2d(32, 64, 3, padding = 1)
        self.norm2_1 = nn.BatchNorm2d(64)
    
        self.conv2_2 = nn.Conv2d(64, 128, 3, padding = 1)
        self.norm2_2 = nn.BatchNorm2d(128)
    
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.dropout2 = nn.Dropout2d(0.25)
        
        # fully connected (single) to RELU
        
        self.fc1 = nn.Linear(128 * 7 * 7, 128)
        self.normfc_1 = nn.BatchNorm1d(128)    
        self.dropoutfc_1 = nn.Dropout1d(0.20)
        
        self.fc2 = nn.Linear(128, 10)
        
        
    def forward(self, x):
        out = self.norm1_1(F.relu(self.conv1_1(x)))
        out = self.norm1_2(F.relu(self.conv1_2(out)))
        out = self.pool1(out)
        out = self.dropout1(out)
        
        out = self.norm2_1(F.relu(self.conv2_1(out)))
        out = self.norm2_2(F.relu(self.conv2_2(out)))
        out = self.pool2(out)
        out = self.dropout2(out)
        
        # flatten
        out = out.view(-1, 128 * 7 * 7)
        
        out = self.normfc_1(F.relu(self.fc1(out)))
        out = self.dropoutfc_1(out)
        
        out = self.fc2(out)
        
        # softmax classifier
        
        return out
