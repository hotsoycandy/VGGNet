"""VGG Train"""

import time
import datetime
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as torchTrans
from torchvision import datasets
from matplotlib import pyplot as plt
from VGGNet import VGGNet
from get_device import get_device

def get_FashionMNIST_datas (batch_size):
    """
    get FashionMNIST datas as DataLoader
    """

    trans = torchTrans.Compose([
        torchTrans.Resize((224, 224)),
        torchTrans.ToTensor()
    ])
    training_data = datasets.FashionMNIST(
        root="Fashion_MNIST_Data",
        train=True,
        download=True,
        transform=trans,
    )
    test_data = datasets.FashionMNIST(
        root="Fashion_MNIST_Data",
        train=False,
        download=True,
        transform=trans,
    )
    train_dataloader = DataLoader(
        training_data,
        batch_size = batch_size,
        shuffle = True
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size = batch_size,
        shuffle = False
    )

    def get_label (idx):
        return training_data.classes[idx]

    return training_data, test_data, train_dataloader, test_dataloader, 1, get_label

def print_imgs (train_dataloader, get_label):
    for _, (X, y) in enumerate(train_dataloader):
        for (X_i, y_i) in zip(X, y) :
            img = X_i.numpy().transpose(1, 2, 0)
            plt.title(get_label(y_i.item()))
            plt.imshow(img)
            plt.show()

def train (dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)

    # epoch
    for batch, (X, y) in enumerate(dataloader) :
        X, y = X.to(device), y.to(device)

        # forward propagation
        pred = model(X)
        loss = loss_fn(pred, y)

        # print loss every 10th learning
        if batch % 10 == 0:
            current = (batch) * len(X)
            print(f"Batch{batch:>5d}. loss: {loss:>7f} [{current:>7d}/{size:>7d}]")

            # backward propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def test (dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def get_time ():
    return datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")

if __name__ == '__main__':
    batch_size = 64
    epochs = 50
    learning_rate = 1e-2
    weight_decay = 1e-4 * 5
    momentum = 0.9

    device = get_device()
    device_count = torch.cuda.device_count()
    print(f"Using {device} device. Cuda device count: {device_count}")

    _, _, train_dataloader, test_dataloader, in_channels, get_label = get_FashionMNIST_datas(batch_size)

    # print_imgs(train_dataloader, get_label)

    model = VGGNet(in_channels).to(device)
    # model.load_state_dict(torch.load("model.pth"))

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr = learning_rate,
        momentum = momentum,
        weight_decay = weight_decay
    )

    print(f"[{get_time()}] Start Training...")
    for t in range(epochs):
        print(f"[{get_time()}] Epoch [{t+1}/{epochs}]\n{'-' * 50}")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print(f"[{get_time()}] Done!")

    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")
