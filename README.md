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

overview, introduce method, reward shaping, experiments
sections
  method
  reward shaping
  experiments

Tasks requiring tightly coupled coordinated behaviors between many agents is difficult for agents to learn due to positive feedback only occurring when multiple agents simultaneously act in a coordinated environment. From the perspective of a learning agent, this would require multiple random sequences of actions to happen to achieve the task, or some part of the task that is enough to provide a positive feedback signal from the environment. Particularly as the number of required agents grows, this becomes exponentially more unlikely with every added agent.

This work introduces a leader-follower paradigm as a method for addressing this necessity of agents having to randomly discover a set of coordinated behaviors for tightly coupled tasks requiring many agents. The method splits agents into two types: leaders and followers. Leaders take on the form of typical learning agents that take the state as input and produce an action as output at every time step. Followers have the same state and action spaces as the leaders, but instead they use a simple preset policy that causes them to move towards nearby agents while maintaining a minimal distance between each other.

[//]: # (introduce the actual task)

The key insight here is that the follower policy acts as a method of injecting domain knowledge about the task without fully specifying the behavior of the system. In a tightly coupled problem, multiple agents must work in close coordination to accomplish the task. The follower policy pushes some agents towards acting in a manner that is conducive to the agents working closely. Often, designers will shape the fitness functions to try and capture how well a task is performed, and this fitness shaping is what is meant to drive the manifestation of a desired behavior. However, simple policies themselves can also serve as an effective means of guiding systems of agents to coordinate in complex manners.

[//]: # (shape the behavior, not the reward)

### Leader-Follower Credit Assignment

While the leader-follower paradigm is able to guide agents to establishing and maintaining coordination in complex tasks, there remains a problem of effectively assigning credit to the leaders so that they can learn to optimize the solution. The problem of credit assignment exists in any multiagent learning problem, especially in the case of episodic rewards. In these cases, it can be extremely difficult to tease out an individual agent's contribution to the task. With the leader-follower paradigm, this problem can become exacerbated as a leader's actions are also responsible for the actions of nearby followers.

We extend the idea of difference rewards to capture this larger impact leaders have on the system.

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
