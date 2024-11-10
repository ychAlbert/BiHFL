# BiHFL
Official code repository for the paper BiHFL: A Biologically Inspired Hierarchical Framework for Federated Continual Learning


### Installation Steps

1. **Install Dependencies**
   After installing PyTorch, you need to install additional dependencies. Run the following commands to install the required packages:

   ```bash
   pip install scikit-learn
   pip install progress
   pip install tensorboardX
   ```
## Quick Start

### Multi-Client Learning with Replay and HLOP
To perform multi-client learning using replay and HLOP, run the following command:
```python
python main.py --experiment_name pmnist --fed_algorithm FedAvg --use_replay --use_hlop --n_client 3
```

### Multi-Client Learning without Replay and HLOP
To perform multi-client learning without using replay and HLOP, run the following command:
```python
python main.py --experiment_name pmnist --fed_algorithm FedAvg --num_clients 3
```

## Using tensorboard to visualize the results
To visualize the results, you can use TensorBoard. First, install TensorBoard by running the following command:
```bash
tensorboard --logdir=D:\BiHFL\logs\pmnist
```