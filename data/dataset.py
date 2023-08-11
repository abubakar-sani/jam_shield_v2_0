#!/usr/bin/env python
# coding:utf-8
"""
name        : dataset.py,
version     : 1.0.0,
url         : https://github.com/abubakar-sani/RFF,
license     : MIT License,
copyright   : Copyright 2021 by Abubakar Sani Ali, Khalifa University,
author      : Abubakar Sani Ali,
email       : engrabubakarsani@gmail.com,
date        : 9/5/2022,
description : Dataset module that returns the processed data,
"""

import math
import os
import numpy as np
import pandas as pd

import json

base_dir = os.path.abspath('')
raw_dir = f'{base_dir}/datasets/raw/spectral_scans_QC9880_ht20_background'
processed_dir = f'{base_dir}/datasets/processed/spectral_scans_QC9880_ht20_background'
    
    
def load_spectral_scans(jammer, jammed_freq, jammer_dist, jamming_power, channels, n_features, n_scans=10):
    if jammer == 'combined':
        scenario = f'samples_chamber_{jammer_dist}cm_{jamming_power}dBm'
    elif jammer == 'none':
        interference = np.random.choice(['high', 'medium', 'low'], p=[0.4, 0.4, 0.2])
        if interference == 'high':
            scenario = f'samples_office_None'
        elif interference == 'low':
            scenario = f'samples_lab_None'
        else:
            scenario = f'samples_chamber_None'
    else:
        scenario = f'samples_chamber_{jammed_freq}MHz_{jammer_dist}cm_{jamming_power}dBm'
    # Process the dataset and generate the raw numpy (to processing it every run)
    if not os.path.isfile(f'{processed_dir}/{scenario}.npy'):
        spectral_data = []
        for channel in channels:
            channel_data = np.empty((0, n_features), int)
            for scan in range(n_scans):
                if jammer == 'combined':
                    scenario = f'samples_chamber_{channel}MHz_{jammer_dist}cm_{jamming_power}dBm'

                df = pd.read_csv(f'{raw_dir}/{scenario}_{scan}.csv')
                temp_data = df.where(df['freq1'] == channel).dropna()
                temp_data = temp_data.to_numpy()
                channel_data = np.append(channel_data, temp_data, axis=0)

            spectral_data.append(channel_data)

        if not os.path.exists(processed_dir):
            os.makedirs(processed_dir)
        if jammer == 'combined':
            scenario = f'samples_chamber_{jammer_dist}cm_{jamming_power}dBm'
        np.save(f'{processed_dir}/{scenario}', np.array(spectral_data, dtype='object'))

    spectral_data = np.load(f'{processed_dir}/{scenario}.npy', allow_pickle=True)

    return spectral_data


def get_feats_dirs(scenario, scan=1):
    spectral_df = pd.read_csv(f'{raw_dir}/{scenario}_{scan}.csv')
    features = spectral_df.columns
    n_features = len(features)
    return features, n_features


def process_data(data, channels, length, stride, time_step, mode=0):
    if mode == 0:
        min_len = len(data[0])
        for i in range(len(data)):
            min_len = len(data[i]) if len(data[i]) < min_len else min_len
        t_samples = min_len
    else:
        t_samples = len(data)
    n_samples = math.floor((t_samples - length) / stride)
    if mode == 1:
        # Breaking data into batches
        data = data[0:t_samples]
        batches = []
        for sample in range(n_samples):
            batch = []
            for i in range(length):
                batch_sample = data[(sample * stride) + i]
                batch.append(batch_sample)
            batches.append(batch)

    else:
        batches = []
        for sample in range(n_samples):
            batch = []
            for i in range(len(channels)):
                channel_data = data[i]
                channel_samples = channel_data[(sample * stride):(length + (sample * stride)), :]
                batch.append(channel_samples)
            batches.append(batch)

    processed_data = np.array(batches[time_step % len(batches)])

    return processed_data
