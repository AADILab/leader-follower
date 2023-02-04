"""
@title

@description

"""
import argparse
import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn

from leader_follower import project_properties


def linear_stack(n_inputs, n_hidden, n_outputs):
    hidden_size = int((n_inputs + n_outputs) / 2)
    network = nn.Sequential(
        nn.Linear(n_inputs, n_outputs)
    )
    for idx in range(n_hidden):
        network.append(nn.Linear(hidden_size, hidden_size))
    network.append(nn.Linear(hidden_size, n_outputs))
    return network


def linear_layer(n_inputs, n_hidden, n_outputs):
    network = nn.Sequential(
        nn.Linear(n_inputs, n_outputs)
    )
    return network


def linear_relu_stack(n_inputs, n_hidden, n_outputs):
    hidden_size = int((n_inputs + n_outputs) / 2)
    network = nn.Sequential(
        nn.Linear(n_inputs, hidden_size),
        nn.ReLU()
    )

    for idx in range(n_hidden):
        network.append(nn.Linear(hidden_size, hidden_size))
        network.append(nn.ReLU())

    network.append(nn.Linear(hidden_size, n_outputs))
    return network


def load_pytorch_model(model_path):
    model = torch.load(model_path)
    model.eval()
    return model


class NeuralNetwork(nn.Module):

    LAST_CREATED = 0

    def __init__(self, n_inputs, n_outputs, n_hidden=2, network_func=linear_layer):
        super(NeuralNetwork, self).__init__()
        self.name = f'{self.LAST_CREATED}'
        self.LAST_CREATED += 1

        self.network_func = network_func

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.flatten = nn.Flatten()
        self.network = self.network_func(n_inputs=n_inputs, n_hidden=n_hidden, n_outputs=n_outputs)
        return

    def __repr__(self):
        return f'{self.name}'

    def copy(self):
        new_copy = copy.copy(self)
        self.LAST_CREATED += 1
        new_copy.name = f'{self.LAST_CREATED}'
        return new_copy

    def device(self):
        dev = next(self.parameters()).device
        return dev

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        if x.dtype is not torch.float32:
            x = x.float()

        if x.shape[0] != self.n_inputs:
            # if input does not have the correct shape
            # x = torch.zeros([1, self.n_inputs], dtype=torch.float32)
            raise ValueError(f'Input does not have correct shape: {x.shape=} | {self.n_inputs=}')

        logits = self.network(x)
        return logits

    def save_model(self, save_dir=None, tag=''):
        # todo optimize saving pytorch model
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training
        if save_dir is None:
            save_dir = project_properties.output_dir
            save_dir = Path(save_dir, 'models')

        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)

        if tag != '':
            tag = f'_{tag}'

        save_name = Path(save_dir, f'{self.name}_model{tag}.pt')
        torch.save(self, save_name)
        return save_name


def main(main_args):
    n_inputs = 4
    n_outputs = 3
    n_hidden = 0

    model = NeuralNetwork(n_inputs=n_inputs, n_outputs=n_outputs, n_hidden=n_hidden)
    print(f'Using device: {model.device()}\n'
          f'{model}')

    save_name = model.save_model()
    print(f'Saved PyTorch model state to {save_name}')

    np_vect = np.random.rand(n_inputs)
    pt_vect = torch.from_numpy(np_vect)
    bad_np_vect = np.random.rand(n_inputs)
    bad_py_vect = torch.from_numpy(bad_np_vect)

    output = model(pt_vect)
    print(f'{output=}')

    bad_output = model(bad_py_vect)
    print(f'{bad_output=}')

    test_vects = [
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
    ]
    for vect in test_vects:
        np_vect = np.asarray(vect)
        pt_vect = torch.from_numpy(np_vect)
        output = model(pt_vect)
        print(f'{vect} | {output=}')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
