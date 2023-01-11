"""
@title

@description

"""
import argparse

from leader_follower.boids_env import BoidsEnv


def main(main_args):
    boid_env = BoidsEnv()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
