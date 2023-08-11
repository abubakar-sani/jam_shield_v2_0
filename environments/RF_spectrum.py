#!/usr/bin/env python
# coding:utf-8
"""
name        : RF_spectrum.py,
version     : 1.0.0,
url         : https://github.com/abubakar-sani/Anti-Jam,
license     : MIT License,
copyright   : Copyright 2022 by Abubakar Sani Ali,
author      : Abubakar Sani Ali,
email       : engrabubakarsani@gmail.com,
date        : 9/9/2022,
description : Environment class for the anti-jamming problem,
"""

# %% Loading libraries
import os
import numpy as np
import pandas as pd
import gym
from gym import spaces
from gym.utils import seeding
from random import randint
from matplotlib import pyplot
import matplotlib as mpl
import copy
import random
from data.dataset import load_spectral_scans, process_data, get_feats_dirs


class RfEnvironment(gym.Env):
    environment_name = "Anti Jamming"

    def __init__(self, jammers, jammer_dist, jamming_power, csc, channels, length, stride, n_scans=10):
        self.selected_freq_prob = None
        self.reward = None
        self.jammed_freq = None
        self.jammer = None
        self.sweep_counter = None
        self.channel_data = None
        self.spectral_data = None
        self.state = None
        self.cstime = 0
        self.csc = csc
        self.interference_types = ['high', 'medium', 'low']
        self.jammers = jammers
        self.jammer_dists = [20]
        self.jammer_dist = jammer_dist
        self.jamming_powers = [10]
        self.jamming_power = jamming_power
        self.jammer_bandwidth = 20  # MHz
        self.n_scans = n_scans
        self.length = length
        self.time_step = 1
        self.stride = stride  # if self.mode == 1 else self.length
        self.channels = channels
        self.freq = self.channels[0]
        self.previous_action = 0
        self.scenario = f'samples_chamber_{self.freq}MHz_{self.jammer_dist}cm_{self.jamming_power}dBm'
        self.features, self.n_features = get_feats_dirs(self.scenario)
        self.stat_features = ['mean']

        self.average_rssi = 0

        self.action_space = spaces.Discrete(len(self.channels))

        self.observation_size = len(self.channels)  # * (1 + (self.n_features - 1) * len(self.stat_features))
        self.observation_space = spaces.Box(low=np.ones(self.observation_size) * -100,
                                            high=np.ones(self.observation_size) * 100)
        # self.seed()
        self.reward_RF = 1
        self.reward_BC = 0.01
        self.n_collisions = 0
        self.trials = 100
        self._max_episode_steps = 100
        self.reward_threshold = 0.95 * self._max_episode_steps
        self.id = "Dynamic Anti-jamming" if len(jammers) > 1 else f"{self.jammers[0]} Anti-jamming"

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """
        Here we reshuffle our dataset and then obtain the RF spectrum scans at the beginning of an episode

        """
        # Getting data
        self.step_count = 0
        self.jammer = np.random.choice(self.jammers)
        self.jammer_dist = np.random.choice(self.jammer_dists)
        self.jamming_power = np.random.choice(self.jamming_powers)

        if self.jammer == 'constant':
            self.jammed_freq = np.random.choice(self.channels)
        elif self.jammer == 'sweeping':
            self.sweep_counter = 0
            self.jammed_freq = self.freq
        elif self.jammer == 'random':
            self.jammed_freq = np.random.choice(self.channels)
        else:
            self.jammed_freq = self.channels[0]

        self.time_step = np.random.randint(100)  # Introducing randomisation at the start

        self.spectral_data = load_spectral_scans(self.jammer, self.jammed_freq, self.jammer_dist, self.jamming_power,
                                                 self.channels, self.n_features)
        self.state = self.get_state(
            process_data(self.spectral_data, self.channels, self.length, self.stride, self.time_step))

        # self.state = np.eye(len(self.channels))[self.channels.index(self.jammed_freq)]

        return self.state.flatten()

    def step(self, action):
        # Get reward of previous action taken (r_t)
        self.step_count += 1
        self.freq = self.channels[action]
        if action != self.previous_action: self.cstime = 1

        # When the agent transmits and no jammer
        if self.jammer == 'none':
            self.reward = self.get_reward(action)

        if self.freq > (self.jammed_freq + self.jammer_bandwidth/2) or self.freq < (self.jammed_freq - self.jammer_bandwidth/2):
            self.reward = self.get_reward(action)
        else:
            # There is collision
            self.reward = 0
            self.n_collisions += 1

        self.previous_action = action

        # Go to next state (s_t+1)
        if self.jammer == 'sweeping':
            self.sweep_counter = self.sweep_counter + 1
            sweep_slot = self.sweep_counter % len(self.channels)
            self.jammed_freq = self.channels[sweep_slot]
        elif self.jammer == 'random':
            self.jammed_freq = np.random.choice(self.channels)

        self.spectral_data = load_spectral_scans(self.jammer, self.jammed_freq, self.jammer_dist, self.jamming_power,
                                                 self.channels, self.n_features)

        self.state = self.get_state(
            process_data(self.spectral_data, self.channels, self.length, self.stride, self.time_step))

        self.time_step = self.time_step + 1
        # self.state = np.eye(len(self.channels))[self.channels.index(self.jammed_freq)]
        if self.step_count >= self._max_episode_steps:
            self.done = True
        else:
            self.done = False

        return self.state.flatten(), self.reward, self.done, self.cstime

    def get_state(self, processed_channel_data):
        state = np.zeros((len(self.channels), 1))
        for channel in range(len(self.channels)):
            channel_state = self.construct_state(processed_channel_data[channel])
            # freq = self.channels[channel]
            # if freq > (self.jammed_freq + self.jammer_bandwidth/2) or freq < (self.jammed_freq - self.jammer_bandwidth/2):
            #     inf_data = self.get_interference_data(channel)
            #     channel_state = self.construct_state(inf_data)
            # else:
            #     channel_state = self.construct_state(processed_channel_data[channel])
            state[channel, :] = channel_state
        return state

    def construct_state(self, processed_channel_data):
        df_channel = pd.DataFrame(processed_channel_data, columns=self.features)
        df_channel[df_channel['snr'] < -100] = np.NaN
        df_channel[df_channel['snr'] > 100] = np.NaN
        df_channel.fillna(df_channel['snr'].mean(), inplace=True)
        state = df_channel['snr'].mean()
        return state

    def get_interference_data(self, channel):
        jammer = 'none'
        inf_spectral_data = load_spectral_scans(jammer, self.jammed_freq, self.jammer_dist, self.jamming_power,
                                                self.channels, self.n_features)
        data_index = np.where(np.array(self.channels) == np.array(self.channels)[channel])
        inf_channel_data = inf_spectral_data[data_index[0][0]]
        inf_data = process_data(inf_channel_data, self.channels, self.length, self.stride, self.time_step, 1)
        return inf_data

    def get_reward(self, action):
        # Penalize agent for switching channel if the channel is not jammed
        return self.reward_RF * (1 - self.csc) if action != self.previous_action else self.reward_RF

    def get_score_to_win(self):
        return self.reward_threshold
