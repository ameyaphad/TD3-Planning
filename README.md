# Autonomous Navigation of TurtleBot3 using Reinforcement Learning

This repository contains code for performing autonomous navigation of a turtlebot in an environment with static obstacles. The Twin Delayed Deep Deterministic Policy Gradient or TD3 algorithm was utilized to train the agent, which was simulated in the Gazebo simulation environment.

## Overview
This project aimed to train a [TurtleBot3](https://emanual.robotis.com/docs/en/platform/turtlebot3/simulation/#gazebo-simulation) robot to navigate between a fixed start position and random goal positions in a confined environment with static obstacles. The simulation was built using **ROS2** and **Gazebo**, with sensor data provided by TurtleBot3 simulation.

---

## Features
- **Robot Platform**: TurtleBot3 in Gazebo simulation.
- **Algorithm**: Twin Delayed Deep Deterministic Policy Gradient (TD3).
- **Environment**: Custom environment file with essential functions (`step()`, `reset()`, etc.) for RL training.

---

## Results
The algorithm successfully trained the agent to perform autonomous navigation with a test accuracy of 70%. The agent was able to carefully navigate between obstacles, proving the effectiveness of the algorithm in continuous environments.

The visualization below shows the turtlebot navigating between a particular set of start and goal positions.

---
