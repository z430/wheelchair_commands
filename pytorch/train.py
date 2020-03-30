import input_data
import pandas as pd
import tensorflow as tf
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 23 * 1, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 8)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 23 * 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main():
    wanted_words = 'left,right,forward,backward,stop,go'
    speech_feature = 'mfcc'
    dataset = input_data.AudioDataset(wanted_words, speech_feature)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)

    # model initialization
    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # train
    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            # print(inputs.shape, labels)
            optimizer.zero_grad()

            outputs = net(inputs.float())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 20 == 0:
                print(f"epoch: {epoch} loss: {(epoch + 1, i + 1, running_loss / 4)}")
                running_loss = 0.0


if __name__ == "__main__":
    main()
    # train, test = tf.keras.datasets.fashion_mnist.load_data()

    # images, labels = train
    # images = images/255.0
    # labels = labels.astype(np.int32)
    # print(labels)
