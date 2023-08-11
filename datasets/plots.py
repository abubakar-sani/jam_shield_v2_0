#!/usr/bin/python

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import time

warnings.filterwarnings("ignore", category=UserWarning)


if __name__ == "__main__":
    base_dir = os.path.abspath('')
    jammed_freq = 2442
    jammer_dist = 20
    jamming_power = 10
    raw_dir = f'{base_dir}/jam_shield/datasets/raw/spectral_scans_QC9880_ht20_background'
    scenario = f'samples_chamber_{jammed_freq}MHz_{jammer_dist}cm_{jamming_power}dBm'
    # scenario = 'samples_chamber_None'
    df = pd.read_csv(f'{raw_dir}/{scenario}_5.csv')
    # Getting channel information
    band = '2.4GHz'
    all_channels = None
    if band == '2.4GHz':
        all_channels = list(range(2412, 2467, 5))
    elif band == '5GHz':
        all_channels = [list(range(5180, 5340, 20)), list(range(5745, 5865, 20))]
        all_channels = [freq for band in range(len(all_channels)) for freq in all_channels[band]]
    else:
        print('Invalid band')

    x = []
    y = []
    for channel in all_channels:
        df_channel = df.where(df['freq1'] == channel).dropna()
        df_channel[df_channel['snr'] < -100] = np.NaN
        df_channel[df_channel['snr'] > 100] = np.NaN
        df_channel.fillna(df_channel['snr'].mean(), inplace=True)
        x.append(channel)
        y.append(df_channel['snr'].mean())

    plotName = f'{scenario}.png'
    plt.bar(x, y, width=4.5)
    plt.xlabel('Frequency (MHz')
    plt.ylabel('Average received power (dBm)')
    plt.savefig(plotName, bbox_inches='tight')
    plt.show()
