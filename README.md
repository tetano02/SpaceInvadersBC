# CTRLC-CTRLPAC - Interactive Imitation Learning Project

## Overview
This project is part of the **Social Robotics – MU5EEH15 (2025/2026)** course. 
It focuses on developing an **interactive imitation learning framework**, combining **Behavioral Cloning** and **DAgger**, aligned with the course objective of exploring human-in-the-loop reinforcement learning.

## Project Description
Our initial plan was to use the **Pac-Man (ALE/Pacman-v5)** environment from Gymnasium, as it provides a visually intuitive and partially stochastic scenario (due to ghost movement).  
However, we realized that the **state space of Pac-Man is highly complex and high-dimensional**, making it challenging to collect and manage demonstrations effectively in our current setup.  

To simplify training and facilitate faster experimentation, we decided to switch to the **Space Invaders (ALE/SpaceInvaders-v5)** environment.  
This environment maintains visual control and temporal dynamics similar to Pac-Man, while offering a **more manageable observation space**, allowing us to focus on the imitation learning aspects rather than extensive state abstraction.

## Objectives
- Implement **Behavioral Cloning** to train an initial policy from human demonstrations (via keyboard teleoperation).  
- Extend the model with **DAgger (Dataset Aggregation)** to iteratively improve policy performance using interactive corrections.  
- Evaluate the effectiveness of human-in-the-loop training in a visual control task.

## Methodology
1. **Data Collection:** Human demonstrations recorded via Gymnasium environment wrappers.  
2. **Behavioral Cloning:** Train a neural policy network to imitate human trajectories.  
3. **DAgger Iterations:** Run interactive sessions where a human provides corrective actions as the agent plays.  
4. **Evaluation:** Compare performance before and after DAgger corrections.

## Tools and Frameworks
- **Gymnasium (Farama Foundation)** – Environment simulation  
- **Stable-Baselines3 / PyTorch** – Reinforcement learning backbone  
- **Minari** – Dataset handling for offline RL and imitation learning  
- **Custom Keyboard Teleoperation Script** – For demonstration collection  

## Expected Results
We expect the agent trained with DAgger to outperform the Behavioral Cloning baseline, particularly in handling previously unseen game states and improving robustness.

## Group Information
**Group Name:** CTRL+C - CTRL+PAC
**Members:** Agnelli Stefano, Cremonesi Andrea, Mombelli Tommaso, Sun Wen Wen



