import torchvision.transforms as transforms
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim

BATCH_SIZE = 256

def get_cifar10_dataloaders():
    cifar10_mean = [0.4914, 0.4822, 0.4465]
    cifar10_std = [0.2023, 0.1994, 0.2010]

    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(cifar10_mean, cifar10_std)])

    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(cifar10_mean, cifar10_std)])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=2)

    return trainloader, testloader


def evaluate_model(model, dataloader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = images.cuda(
                device=device), labels.cuda(device=device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy


def train_model(model, train_dataloader, test_dataloader, num_epochs, device):

    losses = []
    accuracies = []

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01,
                          momentum=0.9, weight_decay=1e-4, nesterov=True)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.01, steps_per_epoch=len(train_dataloader), epochs=num_epochs)

    model.cuda(device=device)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_dataloader):
            inputs, labels = data
            inputs, labels = inputs.cuda(
                device=device), labels.cuda(device=device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            if i % 50 == 49:
                losses.append(running_loss / 50)
                running_loss = 0.0

        if epoch % 1 == 0:
            accuracy = evaluate_model(model, test_dataloader, device)
            accuracies.append(accuracy)

    results = {}
    results['losses'] = losses
    results['accuracies'] = accuracies

    return model, results
