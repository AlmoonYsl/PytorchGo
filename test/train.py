# coding: utf-8
import time
import torch.utils.data
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from pytorch_lib.le_net import *
import matplotlib.pyplot as plt
import numpy as np


# load dataset
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])

# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=False, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
#                                               shuffle=True, num_workers=2)
#
# testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                        download=False, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=4,
#                                              shuffle=False, num_workers=2)
#
# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                      download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                         shuffle=True, num_workers=2)
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                      download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

# classes = ('A', 'B', 'C', 'D', 'E', 'F', 'G',
#            'H', 'I', 'J', 'K', 'L', 'M', 'N',
#            'O', 'P', 'Q', 'R', 'S', 'T', 'U',
#            'V', 'W', 'X', 'Y', 'Z')

classes = ('0', '1', '2', '3', '4', '5', '6',
           '7', '8', '9')


# gpu settings
is_gpu = True
device = torch.device("cuda:0" if torch.cuda.is_available() and is_gpu else "cpu")


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.cpu().numpy()
    plt.figure(3)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    net = LeNet()
    net = net.to(device)
    loss_list = []
    corrects_list = []
    mini_batches = 3000
    print("Using " + str(device))
    print('Start Training')
    start = time.time()
    for epoch in range(2):
        running_loss = 0.0
        running_corrects = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            criterion = mod.CrossEntropyLoss()
            optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
            optimizer.zero_grad()

            # forward
            outputs = net(inputs)
            # backward + optimize
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs, 1)
            running_corrects += (torch.sum(predicted == labels.data).true_divide(4))
            running_loss += loss.item()
            if i % mini_batches == mini_batches - 1:
                print('Finished ' + str(epoch + 1) + 'epoch ' + str(i + 1) + 'mini_batches')
                loss_list.append(running_loss / mini_batches)
                corrects_list.append(running_corrects / mini_batches)
                running_loss = 0.0
                running_corrects = 0.0
    print('Finished Training')
    end = time.time()
    print('Training time:' + str(end - start) + 's')

    plt.figure(1)
    plt.title('Loss')
    plt.plot(loss_list)
    plt.figure(2)
    plt.title('Accuracy')
    plt.plot(corrects_list)

    # save net
    torch.save(net, './net/net.pkl')
    # save net parameters
    torch.save(net.state_dict(), './net/net_params.pkl')

    # test
    dataiter = iter(testloader)
    images, labels = dataiter.__next__()
    images, labels = images.to(device), labels.to(device)
    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
