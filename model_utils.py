import torch
import torch.nn as nn


def get_cifar_vgg(num_labels, pretrained=True):
    model = torch.hub.load('pytorch/vision:v0.6.0',
                           'vgg16', pretrained=pretrained)
    removed = list(model.classifier.children())[:-1]
    model.classifier = nn.Sequential(
        *(removed + [nn.Linear(4096, num_labels)]))
    return model
