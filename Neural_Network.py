import urllib
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from numpy.distutils.conv_template import header
from torchvision import datasets, transforms

def get_data_loader(training = True):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_set = datasets.MNIST(root = 'data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root = 'data', train=False, download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=50)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=50, shuffle=False)
    
    if training:
        return train_loader
    else:
        return test_loader

def build_model():

    warnings.filterwarnings('ignore')
    model = nn.Sequential(nn.Flatten(), nn.LazyLinear(128), nn.ReLU(), nn.LazyLinear(64), nn.ReLU(), nn.LazyLinear(10))
    return model

def train_model(model, train_loader, criterion, T):

    op = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()

    for epoch in range(T):
        running_loss = 0.0
        total = 0
        correct = 0
        for data in train_loader:
            inputs, labels = data
            op.zero_grad()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            loss.backward()
            op.step()
            running_loss += loss.item()
        print("Train Epoch: %d   Accuracy: %d/%d(%.2f%%) Loss: %.3f" %(epoch, correct, total, 100 * (correct / total), running_loss / len(train_loader)))


def evaluate_model(model, test_loader, criterion, show_loss=True):

    model.eval()
    running_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    if show_loss:
        print("Average Loss: %.4f" %(running_loss / len(test_loader)))
        print("Accuracy: %.2f%%" %(100 * (correct / total)))
    else:
        print("Accuracy: %.2f%%" % (100 * (correct / total)))

def predict_label(model, test_images, index):

    final_prob = 0
    class_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    p_list = list()
    p_dict = {}
    output = model(test_images)
    prob = F.softmax(output, dim=1)
    img = list(prob[index])
    predicted = sorted(img)[::-1]
    for tensor in img:
        p_list.append(tensor.item() * 100)
    zero, one, two, three, four, five, six, seven, eight, nine = p_list[0], p_list[1], p_list[2], p_list[3], p_list[4], p_list[5], p_list[6], p_list[7], p_list[8], p_list[9]
    for p in range(3):
        if predicted[p].item() * 100 == zero:
            print("zero: %.2f%%" %(predicted[p].item() * 100))
            continue
        if predicted[p].item() * 100 == one:
            print("one: %.2f%%" %(predicted[p].item() * 100))
            continue
        if predicted[p].item() * 100 == two:
            print("two: %.2f%%" %(predicted[p].item() * 100))
            continue
        if predicted[p].item() * 100 == three:
            print("three: %.2f%%" %(predicted[p].item() * 100))
            continue
        if predicted[p].item() * 100 == four:
            print("four: %.2f%%" %(predicted[p].item() * 100))
            continue
        if predicted[p].item() * 100 == five:
            print("five: %.2f%%" %(predicted[p].item() * 100))
            continue
        if predicted[p].item() * 100 == six:
            print("six: %.2f%%" %(predicted[p].item() * 100))
            continue
        if predicted[p].item() * 100 == seven:
            print("seven: %.2f%%" %(predicted[p].item() * 100))
            continue
        if predicted[p].item() * 100 == eight:
            print("eight: %.2f%%" %(predicted[p].item() * 100))
            continue
        if predicted[p].item() * 100 == nine:
            print("nine: %.2f%%" %(predicted[p].item() * 100))
            continue

if __name__ == '__main__':
    criterion = nn.CrossEntropyLoss()
    model = build_model()
    train_model(model, get_data_loader(), criterion, T=5)
    evaluate_model(model, get_data_loader(training=False), criterion, show_loss=True)
    test_images, _ = iter(get_data_loader()).next()
    predict_label(model, test_images, 1)
