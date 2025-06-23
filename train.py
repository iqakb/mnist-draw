import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import sklearn.metrics as metrics

import os
import cv2 as cv
import numpy as np

BATCH_SIZE = 32

## transformations
transform = transforms.Compose([transforms.ToTensor()])

torch.manual_seed(42)
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=2)

class DigitRecognition(nn.Module):
    def __init__(self):
        super(DigitRecognition, self).__init__()
        self.conv1 = nn.Conv2d(1,32,3)
        self.d1 = nn.Linear(26 * 26 * 32, 128)
        self.d2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        x = self.conv1(x)
        x = F.relu(x)
        
        x = x.flatten(start_dim=1)

        x = self.d1(x)
        x = F.relu(x)

        x = self.d2(x)
        x = F.softmax(x,dim=1)
        return x

learning_rate = 0.001
num_epochs = 5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = DigitRecognition()
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    train_running_loss = 0.0
    train_acc = 0.0
    
    for batch_idx, (images, labels) in enumerate(trainloader):


        images = images.to(device)
        labels = labels.to(device)

        pred = model(images)
        # print(pred)
        loss = criterion(pred,labels)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        
        train_running_loss += loss.detach().item()
        train_acc += (torch.argmax(pred, 1).flatten() == labels).type(torch.float).mean().item()

        # break  # remove this if you want to loop over the entire dataset
    
    print('Epoch: %d | Loss: %.4f | Train Accuracy: %.2f' \
          %(epoch, train_running_loss / batch_idx, train_acc/batch_idx))
    
test_acc = 0.0
for i, (images, labels) in enumerate(testloader, 0):
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
    test_acc += (torch.argmax(outputs, 1).flatten() == labels).type(torch.float).mean().item()
    preds = torch.argmax(outputs, 1).flatten().cpu().numpy()
        
print('Test Accuracy: %.2f'%(test_acc/i))

torch.save(model.state_dict(), './mnist/model_weights.pth')