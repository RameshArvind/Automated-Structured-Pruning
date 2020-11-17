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
for i in trange(1000):
    net.load_state_dict(torch.load("original_trained_network.pt"))
    model_copy = get_cifar_vgg(10, pretrained=False)
    strategy = np.random.uniform(
        0, 0.8, size=num_prunable_layers(model_copy)).tolist()
    results_file_name = f"{str(uuid.uuid1())}.json"
    try:
        model_copy, num_pruned = prune_with_strategy(model_copy, net, strategy)
        pruned_num_parameters = sum(p.numel() for p in model_copy.parameters())
        compression_ratio = original_num_parameters / pruned_num_parameters
        scratch_b_epochs = min(100, round(compression_ratio * EPOCHS))
        model_copy, results = train_model(
            model_copy, trainloader, testloader, scratch_b_epochs, device)
        results['pruned_num_parameters'] = pruned_num_parameters
        results['strategy'] = strategy
        results['did_it_train'] = True
    except:
        results = {}
        results['strategy'] = strategy
        results['did_it_train'] = False
    # https://stackoverflow.com/questions/7100125/storing-python-dictionaries
    with open(results_file_name, 'w') as fp:
        json.dump(results, fp)
    model_copy.cpu()
    del model_copy
