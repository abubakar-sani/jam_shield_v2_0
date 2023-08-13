# Deep Reinforcement Learning Based Anti-Jamming

[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/dwyl/esta/issues)

![RL](utilities/RL_image.jpeg)   ![PyTorch](utilities/PyTorch-logo-2.jpg)

This project implements a deep reinforcement learning-based approach to counteract jamming attacks in wireless communication, as presented in the paper "Defeating Proactive Jammers Using Deep Reinforcement Learning for Resource-Constrained IoT Networks". The implementation employs various variants of Deep Q-Network (DQN) algorithms to train an agent for optimal channel selection, effectively mitigating the impact of jamming attacks. The agent is trained using a custom-generated dataset outlined in the paper's accompanying article [here](https://www.techrxiv.org/articles/preprint/RF_Jamming_Dataset_A_Wireless_Spectral_Scan_Approach_for_Malicious_Interference_Detection/21524508), involving interactions with different types of jammers.

## Getting Started

Follow these instructions to set up and run the project on your local machine for development and testing.

### Prerequisites

- Python 3.7 or higher
- PyTorch
- OpenAI Gym
- Matplotlib
- Numpy
- Pandas

For specific library versions, please refer to the `requirements.txt` file.

### Installation

1. Clone the repository to your local machine.
2. Install the required packages using pip:

   ```bash
   pip install -r requirements.txt
   ```
3. Execute the script:

   ```bash
   python3 results/Anti_Jam.py
   ```

### Usage

The primary script trains different DQN agent variants for a specified number of episodes. After training, the agent's performance is evaluated and plotted. Relevant data, such as agent behavior, rewards, throughput, and channel switching times, are saved for further analysis.

#### Repository Structure

The structure of the repository is designed to maintain clarity and organization:

- **agents**: This directory contains various agent implementations, categorized into different types such as actor-critic, DQN, policy gradient, and stochastic policy search agents.

- **environments**: The directory houses the implementation of the RFSpectrum environment, where the agent operates and learns.

- **results**: This directory stores the data and graphs generated during training and evaluation. The `Anti_Jam.py` script is the main entry point for running the training and evaluation process.

- **tests**: This directory can be used to write and execute tests for the codebase.

- **utilities**: The directory contains utility files, including data structures and visual assets.

#### License

This project is licensed under the MIT License - see the LICENSE.md file for details.

#### Acknowledgements

This project is supported by the following:

- [Deep Reinforcement Learning Algorithms with PyTorch](https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch): This repository provides PyTorch implementations of deep reinforcement learning algorithms and environments.

- **Research Paper**: The implementation is based on the research paper titled "Defeating Proactive Jammers Using Deep Reinforcement Learning for Resource-Constrained IoT Networks". The paper serves as the theoretical foundation for the project and can be accessed [here](https://arxiv.org/abs/2307.06796).

#### Contributing

Contributions to this project are welcome! If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature/fix.
3. Make your changes and commit them with clear messages.
4. Push your changes to your forked repository.
5. Submit a pull request, detailing the changes you made and why they should be merged.

Let's work together to improve this project and make it even more effective in countering jamming attacks!
