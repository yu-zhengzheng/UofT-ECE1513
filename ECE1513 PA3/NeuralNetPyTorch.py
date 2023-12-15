import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
import numpy as np


# Function for loading notMNIST Dataset
def loadData(datafile="notMNIST.npz"):
    with np.load(datafile) as data:
        Data, Target = data["images"].astype(np.float32), data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.0
        Target = Target[randIndx]
        trainData, trainTarget = Data[:10000], Target[:10000]
        validData, validTarget = Data[10000:16000], Target[10000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget


# Custom Dataset class.
class notMNIST(Dataset):
    def __init__(self, annotations, images, transform=None, target_transform=None):
        self.img_labels = annotations
        self.imgs = images
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.imgs[idx]
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


# Define CNN
class CNN(nn.Module):
    def __init__(self, drop_out_p=0.0):
        super(CNN, self).__init__()
        # TODO
        # DEFINE YOUR LAYERS HERE
        # convolutional layer1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4)
        # batchnorm layer1
        self.bn1 = nn.BatchNorm2d(num_features=32)
        # pooling layer1
        self.mp1 = nn.MaxPool2d(kernel_size=(2, 2))
        # convolutional layer2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4)
        # batchnorm layer2
        self.bn2 = nn.BatchNorm2d(num_features=64)
        # pooling layer2
        self.mp2 = nn.MaxPool2d(kernel_size=(2, 2))
        # fully connected layer
        self.FC1 = nn.Linear(1024, 784)
        self.FC2 = nn.Linear(784, 10)
        # dropout layer
        self.D = nn.Dropout(p=drop_out_p)

    def forward(self, x):
        # TODO
        # DEFINE YOUR FORWARD FUNCTION HERE
        # convolutional 1
        x = self.conv1(x)
        # output conv1 operation
        x = F.relu(x)
        # apply the batch norm layer
        x = self.bn1(x)
        # max pooling
        x = self.mp1(x)

        # convolutional 2
        x = self.conv2(x)
        # output conv2 operation
        x = F.relu(x)
        # apply the batch norm layer
        x = self.bn2(x)
        # max pooling
        x = self.mp2(x)

        # flatten operation
        x = torch.flatten(x, start_dim=1)
        # Dropout
        x = self.D(x)
        x = self.FC1(x)
        x = F.relu(x)
        # FC
        x = self.FC2(x)

        return x


# Define FNN
class FNN(nn.Module):
    def __init__(self, drop_out_p=0.0):
        super(FNN, self).__init__()
        # TODO
        # DEFINE YOUR LAYERS HERE

        self.FC1 = nn.Linear(784, 10)
        self.FC2 = nn.Linear(10, 10)
        self.D = nn.Dropout(p=drop_out_p)
        self.FC3 = nn.Linear(10, 10)

    def forward(self, x):
        # TODO
        # DEFINE YOUR FORWARD FUNCTION HERE

        # 1. Flatten
        x = x.flatten(start_dim=1)

        # 2. FC1
        x = self.FC1(x)
        x = F.relu(x)

        # 3. FC2
        x = self.FC2(x)
        x = F.relu(x)

        # 4 . Dropout
        x = self.D(x)

        # 5. FC3
        x = self.FC3(x)

        return x


# Commented out IPython magic to ensure Python compatibility.
# Compute accuracy
def get_accuracy(model, dataloader):
    model.eval()

    device = next(model.parameters()).device
    accuracy = 0.0
    total = 0.0

    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            # TODO
            logits = model(images)
            # compute predicted labels for the batch
            pred = torch.argmax(logits, dim=1)
            # compute accuracy for the batch
            accuracy += sum(pred == labels) / 32

    # compute average accuracy over batches
    accuracy = accuracy / len(dataloader)
    return accuracy


def train(model, device, learning_rate, weight_decay, train_loader, val_loader, test_loader, num_epochs=50, verbose=False):
    # Define your cross entropy loss function here
    # Use cross entropy loss
    criterion = nn.CrossEntropyLoss()

    # Define your optimizer here
    # Use AdamW optimizer, set the weights, learning rate and weight decay argument.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # define the loss function
    # loss = []

    acc_hist = {'train': [], 'val': [], 'test': []}

    for epoch in range(num_epochs):
        model = model.train()
        ## training step
        train_running_loss = 0
        train_acc = 0

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # TODO
            # Follow the step in the tutorial

            ## forward + backprop + loss
            logits = model(images)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()

            ## update model params
            optimizer.step()
            # plot the loss after each update
            # plt.plot(loss)

            # train_running_loss += loss.detach().item()

            # train_acc += get_accuracy2(logits, labels)

        model.eval()
        acc_hist['train'].append(get_accuracy(model, train_loader))
        acc_hist['val'].append(get_accuracy(model, val_loader))
        acc_hist['test'].append(get_accuracy(model, test_loader))

        if ~verbose:
            print('Epoch: %d | Train Accuracy: %.2f | Validation Accuracy: %.2f | Test Accuracy: %.2f' \
                  % (epoch, acc_hist['train'][-1], acc_hist['val'][-1], acc_hist['test'][-1]))

    return model, acc_hist


def experiment(model_type='FNN', learning_rate=0.0001, dropout_rate=0.5, weight_decay=0.01, num_epochs=50, verbose=False):
    # Use GPU if it is available.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Inpute Batch size:
    BATCH_SIZE = 32

    # Convert images to tensor
    transform = transforms.Compose(
        [transforms.ToTensor()])

    # Get train, validation and test data loader.
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()

    train_data = notMNIST(trainTarget, trainData, transform=transform)
    val_data = notMNIST(validTarget, validData, transform=transform)
    test_data = notMNIST(testTarget, testData, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # Specify which model to use
    if model_type == 'CNN':
        model = CNN(dropout_rate)
    elif model_type == 'FNN':
        model = FNN(dropout_rate)

    # Loading model into device
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    model, acc_hist = train(model, device, learning_rate, weight_decay, train_loader, val_loader, test_loader, num_epochs=num_epochs, verbose=verbose)

    # Release the model from the GPU (else the memory wont hold up)
    model.cpu()

    return model, acc_hist


def compare_arch():
    CM, Cacc = experiment(model_type='CNN', learning_rate=0.0001, dropout_rate=0.0, weight_decay=0.0, num_epochs=50, verbose=True)
    FM, Facc = experiment(model_type='FNN', learning_rate=0.0001, dropout_rate=0.0, weight_decay=0.0, num_epochs=50, verbose=True)
    e = np.arange(0, 50, 1)
    plt.plot(e, Cacc['train'], 'r', e, Cacc['test'], 'r.', e, Facc['train'], 'b', e, Facc['test'], 'b.')
    plt.show()


def compare_dropout():
    CM, acc5 = experiment(model_type='CNN', learning_rate=0.0001, dropout_rate=0.5, weight_decay=0.0, num_epochs=50, verbose=True)
    CM, acc8 = experiment(model_type='CNN', learning_rate=0.0001, dropout_rate=0.8, weight_decay=0.0, num_epochs=50, verbose=True)
    CM, acc95 = experiment(model_type='CNN', learning_rate=0.0001, dropout_rate=0.95, weight_decay=0.0, num_epochs=50, verbose=True)
    e = np.arange(0, 50, 1)
    plt.plot(e, acc5['train'], 'r', e, acc5['test'], 'r.', e, acc8['train'], 'b', e, acc8['test'], 'b.', e, acc95['train'], 'g', e, acc95['test'], 'g.')
    plt.show()


def compare_l2():
    CM, acc0 = experiment(model_type='CNN', learning_rate=0.0001, dropout_rate=0.0, weight_decay=0.1, num_epochs=50, verbose=True)
    CM, acc1 = experiment(model_type='CNN', learning_rate=0.0001, dropout_rate=0.0, weight_decay=1.0, num_epochs=50, verbose=True)
    CM, acc10 = experiment(model_type='CNN', learning_rate=0.0001, dropout_rate=0.0, weight_decay=10.0, num_epochs=50, verbose=True)
    e = np.arange(0, 50, 1)
    plt.plot(e, acc0['train'], 'r', e, acc0['test'], 'r.', e, acc1['train'], 'b', e, acc1['test'], 'b.', e, acc10['train'], 'g', e, acc10['test'], 'g.')
    plt.show()


#compare_arch()
compare_dropout()
compare_l2()