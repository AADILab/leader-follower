Leader Follower
=====

# License

See the [LICENSE file](LICENSE) for license rights and limitations (MIT).

# Index

- [Roadmap](#roadmap)
- [Quick State](#quick-start)
  - [Configurations](#configurations)
  - [Experiments](#experiments)
- [Installation](#installation)
  - [Local Package](#local-package)
  - [Pytorch](#pytorch)
- [Testing](#testing)
- [Environment](#environment)
  - [LeaderFollowerEnv](#leaderfollowerenv)
  - [Agents](#agents)
    - [Leader](#leader)
    - [Follower](#follower)
    - [POI](#poi)
  - [Rewards](#rewards)
- [Learning](#learning)
  - [CCEA](#ccea)
    - [Mutation](#mutation)
    - [Selection](#selection)

# Roadmap

<img src="docs/simulator.jpg" alt="simulator" width="250" height="250">

<img src="docs/gap.jpg" alt="gap" width="250" height="250">

<img src="docs/ever_post_gecco.jpg" alt="Ever Post GECCO" width="250" height="250">

## Todo

-[ ] Write draft of approach section
-[x] Figure out why environment is growing at each generation
  - Follower agents were not correctly being reset (not resetting influence and rule histories)
-[x] Script to restart a stat run
-[x] Multiprocessing subpop simulations in each generation
-[ ] Combine multiple stat runs into the same plot
-[ ] Documentation. (Explaining what the parts are and how they work)
-[ ] Create trajectory graph visualization tool (Figure 4 from D++). Include circles around POIs to indicate observation radius.

# Quick Start
## Approach

general overview of approach
	what it is meant to accomplish
	why we should expect this to work for the problem

Inject some piece of information related to the task, not the solution.
Shaping the reward to match the idea of removing the impact of the agent from the system.

## Configurations
## Experiments

# Installation

## Local package

Development mode
```
pip install -e .
```

## Pytorch

# Testing

# Environment

## LeaderFollowerEnv
## Agents
### Leader
### Follower
### POI
## Rewards

# Learning

## CCEA
### Mutation
### Selection
