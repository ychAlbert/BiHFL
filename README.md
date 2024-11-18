# BiHFL
Official code repository for the paper BiHFL: A Biologically Inspired Hierarchical Framework for Federated Continual Learning

## Code Structure

The codebase is structured as follows:

- `FLcore/clients`: Contains client-related modules and scripts.
- `FLcore/dataloader`: Contains data loading scripts for different datasets.
- `FLcore/models`: Contains model definitions.
- `FLcore/modules`: Contains reusable module definitions.
- `FLcore/servers`: Contains server-related modules and scripts.
- `FLcore/utils.py`: Contains utility functions.
- `main.py`: The main entry point for running experiments.
- `main_set.py`: Contains main settings and argument parsing.

### Installation Steps

1. **Install PyTorch**:
   Ensure you have PyTorch installed. You can install it using the official [PyTorch website](https://pytorch.org/get-started/locally/).

2. **Install Additional Dependencies**:
   After installing PyTorch, you need to install additional dependencies. Run the following commands to install the required packages:

   ```bash
   pip install scikit-learn
   pip install progress
   pip install tensorboardX

## Quick Start

### Multi-Client Learning with Allocated Clients' Trainset 
To allocate clients' trainset by dirichlet concentration, run the following command:
```python
python main.py --experiment_name pmnist --fed_algorithm FedAvg --dirichlet
```
To allocate clients' trainset by emd distance, run the following command:
```python
python main.py --experiment_name pmnist --fed_algorithm FedAvg --emd
```
The dirichlet concentration and emd distance of clients both defined in `client_dataset_config.yaml`.

### Multi-Client Learning with Replay and HLOP
To perform multi-client learning using replay and HLOP, run the following command:
```python
python main.py --experiment_name pmnist --fed_algorithm FedAvg --use_replay --use_hlop --dirichlet
```

### Multi-Client Learning without Replay and HLOP
To perform multi-client learning without using replay and HLOP, run the following command:
```python
python main.py --experiment_name pmnist --fed_algorithm FedAvg --dirichlet
```

## Using tensorboard to visualize the results
To visualize the results, you can use TensorBoard. First, install TensorBoard by running the following command:
```bash
tensorboard --logdir=D:\BiHFL\logs\pmnist
```