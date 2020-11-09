### ------ --- Assignment 1 - Question 5 PART A -------- ###
### ----------- 470386390, 470203101, 470205127 -------- ###

# Convolution Neural network 

# Import required packages
import torch.nn as nn
import torch.nn.functional as F 

# Class for the CNN model - it has two convultion layers which use 
# maximum pooling, followed by 2 fully connected layers (Linear)
class convNet(nn.Module):
    def __init__(self):
        super(convNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = (F.max_pool2d(self.conv1(x), 2))
        x = (F.max_pool2d((self.conv2(x)), 2))
        x = x.view(-1, 512)
        x = (self.fc1(x))
        x = F.dropout(x, training=self.training)
        x=  self.fc2(x)
        return x

        

 


