# Automated-Structured-Pruning

## Installation

Clone this repository and run the command `pip install -r requirements.txt`. You are required to be on a linux machine for this install (or the equivalent packages for a windows install) with an NVIDIA GPU.

## Understanding the data / training proxy network

Open the jupyter notebook `Understanding_data.ipynb` and run all the cells. It should reproduce all the graphs utilized in the paper (Except the hyper-parameter exploration graph which can be seen here in this [public google colab notebook](https://colab.research.google.com/drive/1ytgeeDwpnMsm1vibrxBCbCQ966Tu0t4e?usp=sharing)).

## Reproduce / evaluate pruning networks

1. Run `train_base_model.py`. It should produce a file called `original_trained_network.pt` which represents our unpruned base network. This should take roughly 15-20 mins on a machine with GPU.
2. Run `evaluate_pruned_model.py` to create a new model with a defined pruning strategy (taken manually from the proxy network defined in `Understanding_data.ipynb`). It will print the validation accuracy of the pruned model. The `evaluate_pruned_model.py` is essentially a clone of the google colab notebook from before, just made into a python file for easier access. This should take about 1-2 hours on a machine with GPU.