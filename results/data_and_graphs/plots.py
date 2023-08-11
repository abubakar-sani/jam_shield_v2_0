#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Not titled yet
# Author: Abubakar Sani Ali
# GNU Radio version: 3.8.1.0

###################################################################################
# Importing Libraries
###################################################################################
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
import pandas as pd

sns.set_palette('colorblind')

# jammers = ['constant', 'sweeping', 'random', 'dynamic']
jammers = ['dynamic']
agents = ['DQN', 'DQN with Fixed Q Targets', 'DDQN', 'Dueling DDQN', 'DDQN with Prioritised Replay']
network = 'FNN'
history = 1
cscs = [0.0, 0.05, 0.1, 0.15]
max_env_steps = 100
Episodes = 100
tot_iterations = max_env_steps * Episodes

# experiment 1: Convergence time
filename = f'Anti_Jam_training_{cscs[0]}_{jammers[-1]}.pkl'
file = open(filename, 'rb')
object_file = pickle.load(file)
file.close()
n_runs = 3
meanConvergenceTime = []
stdConvergenceTime = []
for agent in agents:
    convergenceTime = []
    for run in range(n_runs):
        agentName = object_file[f'{agent}']
        time = agentName[run][4]
        convergenceTime.append(time)
    meanConvergenceTime.append(np.array(convergenceTime).mean())
    stdConvergenceTime.append(np.array(convergenceTime).std())
print(f'The convergence times are:')
print(meanConvergenceTime)
print(stdConvergenceTime)

# experiment 2: Inference time
# meanInferenceTime = []
# stdInferenceTime = []
# for agent in agents:
#     inferenceTime = []
#     for csc in cscs:
#         filename = f'Anti_Jam_testing_{csc}_{jammer}.pkl'
#         file = open(filename, 'rb')
#         object_file = pickle.load(file)
#         file.close()
#         agentName = object_file[f'{agent}']
#         time = agentName[0][4]
#         inferenceTime.append(time)
#     meanInferenceTime.append(np.array(inferenceTime).mean())
#     stdInferenceTime.append(np.array(inferenceTime).std())
# print(f'The inference times are:')
# print(np.array(meanInferenceTime)/tot_iterations)
# print(np.array(stdInferenceTime)/tot_iterations)
# print(f'The inference speeds are:')
# print(tot_iterations/np.array(meanInferenceTime))
# print(tot_iterations/np.array(stdInferenceTime))

# experiment 3 plots: rewards
for csc in cscs:
    rolling_rewards = np.empty((len(agents), n_runs, Episodes))
    filename = f'Anti_Jam_training_{csc}_{jammers[0]}.pkl'
    file = open(filename, 'rb')
    object_file = pickle.load(file)
    file.close()
    for agent_idx in range(len(agents)):
        agent = agents[agent_idx]
        for run in range(n_runs):
            agentName = object_file[f'{agent}']
            rollingReward = agentName[run][1]
            rolling_rewards[agent_idx][run] = rollingReward

    # Compute the mean and standard deviation of the rolling rewards
    mean_rewards = np.mean(rolling_rewards, axis=1)
    std_rewards = np.std(rolling_rewards, axis=1)

    # Plot the mean rolling rewards and the shaded standard deviation area
    plotName = f'rolling_reward_{csc}_{jammers[0]}.pdf'
    fig, ax = plt.subplots()
    fig.set_figwidth(6)
    fig.set_figheight(5)
    for agent_idx in range(len(agents)):
        ax.plot(mean_rewards[agent_idx], label=f'{agents[agent_idx]}')
        ax.fill_between(range(Episodes), mean_rewards[agent_idx] - std_rewards[agent_idx],
                        mean_rewards[agent_idx] + std_rewards[agent_idx], alpha=0.3)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Rolling Average Reward')

    # Updated legend position
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

    # Adjust the bottom margin to create more space for the legend
    plt.subplots_adjust(bottom=0.25)

    plt.savefig(plotName, bbox_inches='tight')
    plt.show()

# experiment 4 plots: Throughput
for jammer in jammers:
    throughput = []
    for csc in cscs:
        filename = f'Anti_Jam_testing_{csc}_{jammer}.pkl'
        file = open(filename, 'rb')
        object_file = pickle.load(file)
        file.close()
        agentsThroughputs = []
        for agent in agents:
            agentName = object_file[f'{agent}']
            episodeThroughputs = agentName[0][1]
            meanEpisodeThroughput = np.array(episodeThroughputs).mean()
            agentsThroughputs.append(meanEpisodeThroughput)
        throughput.append(agentsThroughputs)
    normalizedThroughput = np.transpose(np.array(throughput) / Episodes)
    X_axis = np.arange(len(cscs))
    plotName = f'throughput_{jammer}.pdf'
    fig, ax = plt.subplots()
    fig.set_figwidth(10)
    fig.set_figheight(4)
    plt.bar(X_axis - 0.3, normalizedThroughput[0], 0.15, label=agents[0])
    plt.bar(X_axis - 0.15, normalizedThroughput[1], 0.15, label=agents[1])
    plt.bar(X_axis + 0, normalizedThroughput[2], 0.15, label=agents[2])
    plt.bar(X_axis + 0.15, normalizedThroughput[3], 0.15, label=agents[3])
    plt.bar(X_axis + 0.3, normalizedThroughput[4], 0.15, label=agents[4])

    plt.ylim((0.6, 1))
    plt.xticks(X_axis, cscs)
    plt.xlabel('Channel switching cost (CSC)')
    plt.ylabel('Normalized Throughput')

    # Updated legend position
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(agents))

    # Adjust the bottom margin to create more space for the legend
    plt.subplots_adjust(bottom=0.25)

    plt.savefig(plotName, bbox_inches='tight')
    plt.show()

# experiment 5 plots: Channel Switching Times
for jammer in jammers:
    cstime = []
    for csc in cscs:
        filename = f'Anti_Jam_testing_{csc}_{jammer}.pkl'
        file = open(filename, 'rb')
        object_file = pickle.load(file)
        file.close()
        agentsCstimes = []
        for agent in agents:
            agentName = object_file[f'{agent}']
            episodeCstimes = agentName[0][-1]
            meanEpisodeCstime = np.array(episodeCstimes).mean()
            agentsCstimes.append(meanEpisodeCstime)
        cstime.append(agentsCstimes)
    normalizedCstime = np.transpose(np.array(cstime) / Episodes)
    X_axis = np.arange(len(cscs))
    plotName = f'cst_{jammer}.pdf'
    fig, ax = plt.subplots()
    fig.set_figwidth(10)
    fig.set_figheight(4)
    plt.bar(X_axis - 0.3, normalizedCstime[0], 0.15, label=agents[0])
    plt.bar(X_axis - 0.15, normalizedCstime[1], 0.15, label=agents[1])
    plt.bar(X_axis + 0, normalizedCstime[2], 0.15, label=agents[2])
    plt.bar(X_axis + 0.15, normalizedCstime[3], 0.15, label=agents[3])
    plt.bar(X_axis + 0.3, normalizedCstime[4], 0.15, label=agents[4])

    plt.ylim((0, 1))
    plt.xticks(X_axis, cscs)
    plt.xlabel('Channel switching cost (CSC)')
    plt.ylabel('Normalized Channel Switiching Frequency')

    # Updated legend position
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(agents))

    # Adjust the bottom margin to create more space for the legend
    plt.subplots_adjust(bottom=0.25)

    plt.savefig(plotName, bbox_inches='tight')
    plt.show()
