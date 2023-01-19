"""
@title

@description

"""
import argparse

import numpy as np

from leader_follower.positions import circle_positions, random_positions, linear_positions
from leader_follower.positions import scale_configuration, translate_configuration, rotate_configuration


def main(main_args):
    np.set_printoptions(precision=3)
    gen_funcs = [
        random_positions,
        linear_positions,
        circle_positions,
    ]
    transform_funcs = [
        scale_configuration,
        translate_configuration,
        rotate_configuration,
    ]
    transform_vals = [0.5, 1, 2]
    num_agents = np.arange(start=1, stop=10, step=1)
    for each_gen in gen_funcs:
        print(f'{each_gen.__name__}')
        positions = []
        for each_val in num_agents:
            pos = each_gen(num_agents=each_val)
            positions.append(pos)
            print(f'\t{each_val=}')
            for row in pos:
                print(f'\t{row}')
            print(f'{"-" * 80}')
    for each_transform in transform_funcs:
        print(f'\t{each_transform.__name__}')
        for each_scale in transform_vals:
            pos = linear_positions(num_agents=10)
            scaled = each_transform(positions=pos, scale=each_scale)
            print(f'\t{each_scale=}')
            for row in scaled:
                print(f'\t{row}')
            print(f'{"-" * 80}')
    # restore default numpy print options
    np.set_printoptions(
        edgeitems=3, infstr='inf', linewidth=75, nanstr='nan', precision=8,
        suppress=False, threshold=1000, formatter=None
    )
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
