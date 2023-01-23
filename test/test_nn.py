"""
@title

@description

"""
import argparse

import numpy as np
import torch

from leader_follower.learn.neural_network import NeuralNetwork


def main(main_args):
    n_inputs = 4
    n_outputs = 2
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
