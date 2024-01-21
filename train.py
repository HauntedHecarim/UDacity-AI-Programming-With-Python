import argparse
from os.path import isdir
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict

def arg_parser():
    parser = argparse.ArgumentParser(description="Train.py")
    parser.add_argument('--arch', dest="arch", action="store", default="vgg16", type=str)
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="./vgg16_bn_checkpoint.pth")
    parser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
    parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=120)
    parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
    parser.add_argument('--gpu', dest="gpu", action="store", default="gpu")
    return parser.parse_args()

def transform_data(data_dir, is_train=True):
    transform_type = transforms.Compose([
        transforms.RandomRotation(30) if is_train else transforms.Resize(256),
        transforms.RandomResizedCrop(224) if is_train else transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip() if is_train else transforms.ToTensor(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data = datasets.ImageFolder(data_dir, transform=transform_type)
    return data

def data_loader(data, is_train=True):
    return torch.utils.data.DataLoader(data, batch_size=50, shuffle=is_train)

def check_gpu(gpu_arg):
    return torch.device("cuda:0" if gpu_arg and torch.cuda.is_available() else "cpu")

def create_model(architecture):
    model = models.__dict__.get(architecture)(pretrained=True)
    model.name = architecture
    for param in model.parameters():
        param.requires_grad = False
    return model

def create_classifier(input_size, hidden_units, output_size):
    return nn.Sequential(OrderedDict([
        ('inputs', nn.Linear(input_size, hidden_units)),
        ('relu1', nn.ReLU()),
        ('dropout', nn.Dropout(0.5)),
        ('hidden_layer1', nn.Linear(hidden_units, output_size)),
        ('relu2', nn.ReLU()),
        ('hidden_layer2', nn.Linear(output_size, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

def validate_model(model, test_loader, device):
    correct, total = 0, 0
    with torch.no_grad():
        model.eval()
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy on test images: {:.2%}'.format(correct / total))

def save_checkpoint(model, save_dir, train_data):
    if save_dir is None or not isdir(save_dir):
        print("Directory not found. Model will not be saved.")
        return

    model.class_to_idx = train_data.class_to_idx
    checkpoint = {
        'architecture': model.name,
        'classifier': model.classifier,
        'class_to_idx': model.class_to_idx,
        'state_dict': model.state_dict()
    }

    torch.save(checkpoint, save_dir)
    print(f"Model checkpoint saved at: {save_dir}")

def train_model(model, train_loader, valid_loader, device, criterion, optimizer, epochs, print_every):
    model.to(device)
    steps = 0

    for e in range(epochs):
        running_loss = 0
        model.train()

        for inputs, labels in train_loader:
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
                    valid_loss, accuracy = validation(model, valid_loader, criterion)
                print(f"Epoch: {e+1}/{epochs} | "
                      f"Training Loss: {running_loss/print_every:.4f} | "
                      f"Validation Loss: {valid_loss/len(valid_loader):.4f} | "
                      f"Validation Accuracy: {accuracy/len(valid_loader):.4f}")
                running_loss = 0
                model.train()

def main():
    args = arg_parser()

    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_data = transform_data(train_dir, is_train=True)
    valid_data = transform_data(valid_dir, is_train=False)
    test_data = transform_data(test_dir, is_train=False)

    train_loader = data_loader(train_data, is_train=True)
    valid_loader = data_loader(valid_data, is_train=False)
    test_loader = data_loader(test_data, is_train=False)

    model = create_model(architecture=args.arch)
    model.classifier = create_classifier(25088, args.hidden_units, 102)

    device = check_gpu(gpu_arg=args.gpu)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    train_model(model, train_loader, valid_loader, device, criterion, optimizer, args.epochs, print_every=30)
    validate_model(model, test_loader, device)
    save_checkpoint(model, args.save_dir, train_data)

if __name__ == '__main__':
    main()
