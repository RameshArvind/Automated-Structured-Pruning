from model_utils import get_cifar_vgg
import torch
from pruning_utils import num_prunable_layers, prune_with_strategy
import uuid
import json
from training_utils import get_cifar10_dataloaders, train_model
import numpy as np
from tqdm.auto import trange

trainloader, testloader = get_cifar10_dataloaders()

device = torch.device(0)
net = get_cifar_vgg(10)
original_num_parameters = sum(p.numel() for p in net.parameters())

EPOCHS = 10

net.load_state_dict(torch.load("original_trained_network.pt"))
model_copy = get_cifar_vgg(10, pretrained=True)
strategy = [0.6194765402050686,
 0.23458329433614009,
 0.4151458469710342,
 0.8734475991929948,
 0.763230170502319,
 0.8098718461487932,
 0.2429063763187106,
 0.7807972186697804,
 0.7213715975315556,
 0.06745635715899885,
 0.5128512347326326,
 0.5289552915111074,
 0.2474687908815043,
 0.46392974188174346,
 0.4533113143780354]

model_copy, num_pruned = prune_with_strategy(model_copy, net, strategy)
pruned_num_parameters = sum(p.numel() for p in model_copy.parameters())
compression_ratio = original_num_parameters / pruned_num_parameters
scratch_b_epochs = 150
num_parameters_pruned = round(100 - 100 * pruned_num_parameters / original_num_parameters, 2)
retained_parameters = round(100 - num_parameters_pruned, 2)

model_copy, results = train_model(model_copy, trainloader, testloader, scratch_b_epochs, device)
print(f"Accuracy of pruned model is {max(results['accuracies'])} %")