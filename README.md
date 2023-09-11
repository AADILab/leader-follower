# Leader-Follower Repository

This repository holds the code for D-Indirect and Agent Specific PBRS for influence based reward shaping. This is specifically designed to address the issue of giving agents in a multiagent system credit not just for their direct contribution to a system objective, but also for how they influence other agents to contribute to that objective. This research is being conducted by the Autonomous Agents and Distributed Intelligence Lab. The leader-follower environment specifically necessitates this type of reward shaping, making it the ideal candidate for this research. This repo includes the code for D-Indirect, Agent Specific PBRS, and several baselines to compare against. The learning algorithm included here is a Cooperative Coevolutionary Algorithm, or a CCEA.

The folders are organized as follows:

- configs: configuration files for different environment settings
- experiments: scripts for generating config files for different experiments
- experiments/tools/run_configs.py: script for actually running the experiments
- figures: folder for saving figures generated from analyzing results
- lib: code for simulator and CCEA
- play: scripts for more interactive and dynamic visuals for simulator and results
- plot: scripts for plotting results
- results: folder for saving results to
- scratch: random scripts for testing different ideas, libraries, or python functions
- tests: tests for different parts of the library

## Install

Go to the base folder of this repository and run

`pip install -e .`

Then import the library in python using

`import leaderfollower`

To get specific classes from different modules, use the following syntax

`from leaderfollower.ccea_lib import CCEA`
