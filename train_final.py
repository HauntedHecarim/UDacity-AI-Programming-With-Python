import argparse
import torch
from collections import OrderedDict
import os
from os.path import isdir
from torch import nn, optim
from torchvision import datasets, transforms, models

def arg_parser():
    parser = argparse.ArgumentParser(description="Train.py")
    parser.add_argument('--arch', dest="arch", action="store", default="vgg16", type=str)
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="checkpoint.pth")
    parser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001, type=float)
    parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=120)
    parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=5)
    parser.add_argument('--gpu', dest="gpu", action="store", default="gpu")
    args = parser.parse_args()
    return args

def train_transformer(train_dir):
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    return train_data

def test_transformer(test_dir):
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    return test_data

def data_loader(data, train=True):
    if train:
        loader = torch.utils.data.DataLoader(data, batch_size=50, shuffle=True)
    else:
        loader = torch.utils.data.DataLoader(data, batch_size=50)
    return loader

def check_gpu(gpu_arg):
    if not gpu_arg:
        return torch.device("cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("CUDA was not found on device, using CPU instead.")
    return device

def primary_loader_model(architecture="vgg16"):
    model = models.vgg16(pretrained=True)
    model.name = "vgg16"
    for param in model.parameters():
        param.requires_grad = False
    return model

def initial_classifier(model, hidden_units):
    classifier = nn.Sequential(OrderedDict([
        ('inputs', nn.Linear(25088, hidden_units)),
        ('relu1', nn.ReLU()),
        ('dropout', nn.Dropout(0.5)),
        ('hidden_layer1', nn.Linear(hidden_units, 90)),
        ('relu2', nn.ReLU()),
        ('hidden_layer2', nn.Linear(90, 70)),
        ('relu3', nn.ReLU()),
        ('hidden_layer3', nn.Linear(70, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    model.classifier = classifier
    return classifier

def validation(model, testloader, criterion, device):
    test_loss = 0
    accuracy = 0
    model.to(device)
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            output = model.forward(inputs)
            test_loss += criterion(output, labels).item()
            ps = torch.exp(output)
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()
    return test_loss, accuracy

def network_trainer(model, trainloader, validloader, device, criterion, optimizer, epochs, print_every, steps):
    model.to(device)
    model.train()
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, validloader, criterion, device)
                print(f"Epoch: {e+1}/{epochs} | "
                      f"Training Loss: {running_loss/print_every:.4f} | "
                      f"Validation Loss: {valid_loss/len(validloader):.4f} | "
                      f"Validation Accuracy: {accuracy/len(validloader):.4f}")
                running_loss = 0
                model.train()
    return model

def validate_model(model, testloader, device):
    correct, total = 0, 0
    with torch.no_grad():
        model.eval()
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy on test images is: {100 * correct / total:.2f}%')

def initial_checkpoint(model, save_dir, train_data):
    if type(save_dir) == type(None):
        print("Model checkpoint directory not specified, model will not be saved.")
    else:
        save_dir = "/home/workspace/ImageClassifier/"
        print("Save Directory:", save_dir)
        if isdir(save_dir):
            model.class_to_idx = train_data.class_to_idx
            torch.save({
                'structure': 'vgg16',
                'hidden_layer1': 4096,
                'dropout': 0.5,
                'epochs': 12,
                'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx,
            }, 'checkpoint.pth')
            model.class_to_idx = train_data.class_to_idx
            checkpoint = {'architecture': model.name,
                          'classifier': model.classifier,
                          'class_to_idx': model.class_to_idx,
                          'state_dict': model.state_dict()}
            torch.save(checkpoint, 'chekpoint.pth')
        else:
            print("Directory not found, model will not be saved.")

def main():
    args = arg_parser()
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    train_data = train_transformer(train_dir)
    valid_data = test_transformer(valid_dir)
    test_data = test_transformer(test_dir)
    trainloader = data_loader(train_data)
    validloader = data_loader(valid_data, train=False)
    testloader = data_loader(test_data, train=False)
    model = primary_loader_model(architecture=args.arch)
    model.classifier = initial_classifier(model, hidden_units=args.hidden_units)
    device = check_gpu(gpu_arg=args.gpu)
    model.to(device)
    if type(args.learning_rate) == type(None):
        learning_rate = 0.001
        print("Learning rate specified as 0.001")
    else:
        learning_rate = args.learning_rate
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    print_every = 30
    steps = 0
    trained_model = network_trainer(model, trainloader, validloader, device, criterion, optimizer, args.epochs, print_every, steps)
    print("\nTraining process is completed!!")
    validate_model(trained_model, testloader, device)
    initial_checkpoint(trained_model, args.save_dir, train_data)

if __name__ == '__main__':
    main()
