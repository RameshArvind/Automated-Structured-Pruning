from training_utils import get_cifar10_dataloaders, evaluate_model, train_model
from model_utils import get_cifar_vgg
import torch

EPOCHS = 13

trainloader, testloader = get_cifar10_dataloaders()
net = get_cifar_vgg(10)
train_model(net, trainloader, testloader, EPOCHS, torch.device(0))
accuracy = evaluate_model(net, testloader, torch.device(0))
print('Accuracy of the network on the 10000 test images: %d %%' % (accuracy))

torch.save(net.state_dict(), "../drive/MyDrive/pruning_results/original_trained_network.pt")
