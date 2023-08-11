import os
import sys
from os.path import dirname, abspath
from pathlib import Path

sys.path.append(dirname(dirname(abspath(__file__))))

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
from utilities.data_structures.Config import Config
from agents.DQN_agents.DQN import DQN
from agents.DQN_agents.DQN_With_Fixed_Q_Targets import DQN_With_Fixed_Q_Targets
from agents.DQN_agents.DDQN import DDQN
from agents.DQN_agents.DDQN_With_Prioritised_Experience_Replay import DDQN_With_Prioritised_Experience_Replay
from agents.DQN_agents.Dueling_DDQN import Dueling_DDQN
from agents.Trainer import Trainer
from environments.RF_spectrum import RfEnvironment

config = Config()
config.seed = 1
# Setup environment
jammers = ['constant']
jammer = 'dynamic' if len(jammers) > 1 else jammers[0]
env_settings = sys.argv
jammer_distance = 20  # cm
jamming_power = 10  # dBm
all_channels = [list(range(5180, 5340, 20))]
all_channels = [freq for band in range(len(all_channels)) for freq in all_channels[band]]
length = 32  # int(sample_rate * sensing_window)
stride = 1
csc = 0.2
env = RfEnvironment(jammers, jammer_distance, jamming_power, csc, all_channels, length, stride)
ob_space = env.features
ac_space = env.action_space
a_size = len(all_channels)
s_size = env.observation_size
config.environment = env

config.base_dir = os.path.abspath('')
config.models_dir = f'{config.base_dir}/models/exp_{csc}_{jammer}'
if not os.path.exists(config.models_dir):
    os.makedirs(config.models_dir)
config.file_to_save_data_results = f"{config.base_dir}/results/data_and_graphs/Anti_Jam_training_{csc}_{jammer}.pkl"
config.file_to_save_results_graph = f"{config.base_dir}/results/data_and_graphs/Anti_Jam_training_{csc}_{jammer}.png"

config.num_episodes_to_run = 100
config.runs_per_agent = 3
config.show_solution_score = False
config.visualise_individual_results = False
config.visualise_overall_agent_results = True
config.standard_deviation_results = 1.0
config.use_GPU = False
config.overwrite_existing_results_file = True
config.randomise_random_seed = True
config.save_model = True
config.training = True

config.hyperparameters = {
    "DQN_Agents": {
        "learning_rate": 0.001,
        "batch_size": 64,
        "buffer_size": 100000,
        "epsilon": 1.0,
        "epsilon_decay_rate_denominator": 1,
        "discount_rate": 0.99,
        "tau": 0.01,
        "alpha_prioritised_replay": 0.6,
        "beta_prioritised_replay": 0.1,
        "incremental_td_error": 1e-8,
        "update_every_n_steps": 1,
        "linear_hidden_units": [256, 256],
        "final_layer_activation": "None",
        "batch_norm": False,
        "gradient_clipping_norm": 0.7,
        "learning_iterations": 1,
        "clip_rewards": False,
        "HER_sample_proportion": 0.8,
        "y_range": (-1, 14)
    },
    "Stochastic_Policy_Search_Agents": {
        "policy_network_type": "Linear",
        "noise_scale_start": 1e-2,
        "noise_scale_min": 1e-3,
        "noise_scale_max": 2.0,
        "noise_scale_growth_factor": 2.0,
        "stochastic_action_decision": False,
        "num_policies": 10,
        "episodes_per_policy": 1,
        "num_policies_to_keep": 5,
        "clip_rewards": False
    },
    "Policy_Gradient_Agents": {
        "learning_rate": 0.05,
        "linear_hidden_units": [256, 256],
        "final_layer_activation": "SOFTMAX",
        "learning_iterations_per_round": 5,
        "discount_rate": 0.99,
        "batch_norm": False,
        "clip_epsilon": 0.1,
        "episodes_per_learning_round": 4,
        "normalise_rewards": True,
        "gradient_clipping_norm": 7.0,
        "mu": 0.0,  # only required for continuous action games
        "theta": 0.0,  # only required for continuous action games
        "sigma": 0.0,  # only required for continuous action games
        "epsilon_decay_rate_denominator": 1.0,
        "clip_rewards": False
    },

    "Actor_Critic_Agents": {

        "learning_rate": 0.005,
        "linear_hidden_units": [256, 256],
        "final_layer_activation": ["SOFTMAX", None],
        "gradient_clipping_norm": 5.0,
        "discount_rate": 0.99,
        "epsilon_decay_rate_denominator": 1.0,
        "normalise_rewards": True,
        "exploration_worker_difference": 2.0,
        "clip_rewards": False,

        "Actor": {
            "learning_rate": 0.0003,
            "linear_hidden_units": [256, 256],
            "final_layer_activation": "Softmax",
            "batch_norm": False,
            "tau": 0.005,
            "gradient_clipping_norm": 5,
            "initialiser": "Xavier"
        },

        "Critic": {
            "learning_rate": 0.0003,
            "linear_hidden_units": [256, 256],
            "final_layer_activation": None,
            "batch_norm": False,
            "buffer_size": 1000000,
            "tau": 0.005,
            "gradient_clipping_norm": 5,
            "initialiser": "Xavier"
        },

        "min_steps_before_learning": 400,
        "batch_size": 64,
        "discount_rate": 0.99,
        "mu": 0.0,  # for O-H noise
        "theta": 0.15,  # for O-H noise
        "sigma": 0.25,  # for O-H noise
        "action_noise_std": 0.2,  # for TD3
        "action_noise_clipping_range": 0.5,  # for TD3
        "update_every_n_steps": 1,
        "learning_updates_per_learning_session": 1,
        "automatically_tune_entropy_hyperparameter": True,
        "entropy_term_weight": None,
        "add_extra_noise": False,
        "do_evaluation_iterations": True
    }
}

if __name__ == "__main__":
    # Training
    AGENTS = [DQN, DQN_With_Fixed_Q_Targets, DDQN, Dueling_DDQN, DDQN_With_Prioritised_Experience_Replay]
    trainer = Trainer(config, AGENTS)
    trainer.run_games_for_agents()
    # Testing
    config.training = False
    config.runs_per_agent = 1
    config.file_to_save_data_results = f"{config.base_dir}/results/data_and_graphs/Anti_Jam_testing_{csc}_{jammer}.pkl"
    config.file_to_save_results_graph = f"{config.base_dir}/results/data_and_graphs/Anti_Jam_testing_{csc}_{jammer}.png"
    trainer = Trainer(config, AGENTS)
    trainer.run_games_for_agents()
