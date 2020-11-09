### ------ --- Assignment 1 - Question 5 PART A -------- ###
### ----------- 470386390, 470203101, 470205127 -------- ###

# ------- Import required packages ----- #
import torch
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image

# ------- Import required functions from file----- #
from neuralNetwork import convNet
from confusionMatrix import plotConfusionMatrix

# ------- Constants----- #
BATCH_SIZE = 100
CLASSES = ('0','1','2','3','4','5','6','7','8','9')

# ------- Load the dataset from the torch package ----- #
def load_datasets():
    set_transform = transforms.Compose([transforms.ToTensor()])

    # Download the training and testing set if it hasn't been already
    # Sort into batches, shuffle and transform them into tensors
    train_set = torch.utils.data.DataLoader(datasets.MNIST('/files/', train=True, download=True,
                                transform=set_transform), batch_size=BATCH_SIZE, shuffle=True)

    test_set = torch.utils.data.DataLoader(datasets.MNIST('/files/', train=False, download=True,
                                transform=set_transform), batch_size=BATCH_SIZE, shuffle=True)

    # Debugging - check that we are obtaining the training and testing set correctly
    train_iter = enumerate(train_set)
    train_batch_idx, (train_images, train_labels)= next(train_iter)

    test_iter = enumerate(test_set)
    test_batch_idx, (test_images, test_labels)= next(test_iter)

    # print(train_batch_idx)
    # print("Training images size: ",train_images.shape)
    # print("Training labels size: ", train_labels.shape)
    return train_images,train_labels, test_images,test_labels,train_set,test_set

# ------- Display images in a 4x4 subplot ----- #
def show_images(images,labels,title=''):
    fig = plt.figure(figsize=(4,4))
    for i in range(len(images)):
        plt.subplot(4,4, i+1)
        plt.imshow(images[i, 0, :, :] * 127.5 + 127.5, cmap='gray')
        plt.title("{}: {}".format(title,labels[i]))
        plt.axis('off')
    plt.show()

# ------- Train the network  ----- #
def train(train_loader,network,epoch):
    # Set the parameters for training
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10

    # Adam optimizer
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    network.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()                           # Start with 0 gradient
        output = network(data)                          # Run through the neural network
        y = torch.zeros(BATCH_SIZE, 10)
        y[range(y.shape[0]), target]=1                  # One hot encoding of the target
        loss = F.binary_cross_entropy_with_logits(output, y)    # Compare the output and target
        loss.backward()                                 # Back propagation
        optimizer.step()

        # Print to screen the epoch and current loss
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

        PATH = './trainedmodel{}.pth'.format(epoch)
        # torch.save(network.state_dict(), PATH)          # Save the model

# ------- Test the network ----- #
def test(test_loader,network):
    # Set parameters for the network
    network.eval()
    correct = 0
    total = 0
    total_pred = []
    total_target = []

    # test_counter = [i*64 for i in range(3 + 1)]

    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)                      # Run through the network
            y = torch.zeros(BATCH_SIZE, 10)
            y[range(y.shape[0]), target]=1              # Convert target to one-hot encoding
            test_loss += F.binary_cross_entropy_with_logits(output, y, reduction='sum')
            _,pred = torch.max(output.data,1)           # Predictions
            total += target.size(0)
            correct += (pred==target).sum().item()      # Number of correct predictions

            total_pred = total_pred + list(pred)        # Store the predicted and targeted
            total_target = total_target +list(target)   # values for confusion matrix


    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, total,100. * correct / total))
    return total_target, total_pred

# ------- Load images from folder ----- #
def loadSampleImgs(folder):
    imgs = torch.Tensor()                           # Convert to Tensor
    for filename in os.listdir(folder):
        filename = '{}{}'.format(folder,filename)   # Collect filename
        # print(filename)
        img = Image.open(filename).convert("L")
        img = np.resize(img, (1,28,28))             # Resize to 1x28x28 image
        im2arr = np.array(img)
        # print(im2arr)
        im2arr = im2arr.reshape(1,1,28,28)
        data = torch.from_numpy(im2arr).float()     # Restructure to torch tensor
        imgs = torch.cat([imgs,data],dim=0)         # Concatenate with previous tensor
        # print(imgs.shape)
    return imgs

# ------- Main program ----- #
if __name__ == '__main__':
    # Load the dataset
    train_X,train_y, test_X,test_y,train_set,test_set = load_datasets()

    # Show some images from the training set
    show_images(train_X[0:16],train_y[0:16],title = 'Ground truth')

    ### ------- Train the model ----- ###
    # # Initialise the convolution neural network
    # net = convNet()
    # # print(net)

    # # Start training
    # for epoch in range(1,4):
    #     train(train_set,net,epoch)

    # # Test the trained network
    # target,pred = test(test_set,net)

    # # Plot the confusion matrix on the testing set
    # plotConfusionMatrix(target,pred,CLASSES)

    ### ------- Retrieve saved model ----- ###
    # Define the neural network
    model = convNet()

    # Load and evaluate the model from file
    model.load_state_dict(torch.load('trainedmodel3.pth'))
    model.eval()

    # Load sample images outside of the MNIST dataset
    folder = './mysamples/'
    data = loadSampleImgs(folder)
    # print(data.shape)

    # Obtain the predictions from the trained network
    output = model(data)

    # Convert predictions to one-hot encoding
    output = torch.softmax(output,dim=1)
    a = np.round(output.detach().numpy(),0)
    y = [np.where(r==1.0)[0] for r in a]
    # print(len(y))

    # Display predictions and images
    show_images(data,y,title='Predictions')
