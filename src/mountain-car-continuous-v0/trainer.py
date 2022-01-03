import os.path
import pickle
from typing import List, Optional
import logging

import gym
import numpy as np
import yaml
from keras.layers import Dense
from keras.models import Sequential, load_model, Model
from keras.optimizer_v2.adam import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, Callback
from tqdm import tqdm

from src.utils.dataset import extend_and_save_arr
from src.utils.gpu import set_memory_growth


class MountainCarContinuousTrainer:

    def __init__(self, config_path="config.yml"):

        with open(config_path) as config_file:
            self.config = yaml.load(config_file, Loader=yaml.FullLoader)

        self.log = logging.Logger("MountainCarContinuousTrainer")

        self.env_config = self.config["env"]
        self.env = gym.make(self.env_config["name"])
        self.max_step: int = self.env_config["max_step"]
        self.observation_size: int = self.env.observation_space.shape[0]

        self.value_config = self.config["value"]
        self.n_warmup_sim: int = self.value_config["n_sim"]
        self.min_pos_episodes: int = self.value_config["min_pos_episodes"]
        self.at_most_n_neg_episodes_per_pos: float = self.value_config["at_most_n_neg_episodes_per_pos"]
        self.extreme_action_chance: float = self.value_config["extreme_action_chance"]
        self.value_scale: float = self.value_config["label_scale"]
        self.min_delta_value = self.value_config["min_delta"]
        self.monitor_value = self.value_config["monitor"]
        self.x_obs_value_arr: List[List[float]] = []
        self.x_action_value_arr: List[List[float]] = []
        self.y_reward_value_arr: List[List[float]] = []
        self.x_obs_value: np.ndarray = np.empty(0)
        self.x_action_value: np.ndarray = np.empty(0)
        self.y_reward_value: np.ndarray = np.empty(0)

        self.eval_config = self.config["eval"]
        self.score_n_simulations = self.eval_config["n_sim"]
        self.render_every_n_step: Optional[int] = self.eval_config["render_every_n_step"]

        self.policy_config = self.config["policy"]
        self.n_policy_sim = self.policy_config["n_sim"]
        self.n_policy_samples = self.policy_config["n_policy_samples"]
        self.sample_n_actions = self.policy_config["n_action_samples"]


    def get_warmup_value_dataset(self):

        self.log.info("Creating warmup value dataset")

        x_obs_arr = []
        x_action_arr = []
        y_reward_arr = []

        pos_episodes = 0
        neg_episodes = 0
        for _ in tqdm(range(self.n_warmup_sim)):

            if pos_episodes >= self.min_pos_episodes:
                break

            observation = self.env.reset()
            total_reward = 0
            t = 0
            observations = []
            actions = []
            for t in range(self.max_step):

                if np.random.random() < self.extreme_action_chance:
                    action = np.array([1.0 if np.random.random() > 0.5 else -1.0])
                else:
                    action = self.env.action_space.sample()
                observations.append(observation)
                actions.append(action)
                observation, reward, done, info = self.env.step(action)
                total_reward += reward
                if done:
                    break

            if total_reward > 0.0:
                x_obs_arr.extend(observations)
                x_action_arr.extend(actions)
                y_reward_arr.extend([[total_reward]] * len(observations))
                pos_episodes += 1
                self.log.debug(f"Successful episode {pos_episodes}/{self.min_pos_episodes} with reward: {total_reward}")
                if pos_episodes >= self.min_pos_episodes:
                    break
            elif neg_episodes < self.at_most_n_neg_episodes_per_pos * (pos_episodes + 1):
                x_obs_arr.extend(observations)
                x_action_arr.extend(actions)
                y_reward_arr.extend([[total_reward]] * len(observations))
                neg_episodes += 1

        self.log.info(f"Successful episodes: {pos_episodes}")

        return x_obs_arr, x_action_arr, y_reward_arr

    def create_value_dataset(self):
        self.x_obs_value_arr, self.x_action_value_arr, self.y_reward_value_arr = self.get_warmup_value_dataset()
        os.makedirs("temp", exist_ok=True)
        extend_and_save_arr(self.x_obs_value_arr, "temp/x_obs_arr.p")
        extend_and_save_arr(self.x_action_value_arr, "temp/x_action_arr.p")
        extend_and_save_arr(self.y_reward_value_arr, "temp/y_reward_arr.p")

    def create_training_data(self):
        self.x_obs_value = np.array(self.x_obs_value_arr)
        self.x_action_value = np.array(self.x_action_value_arr)
        self.y_reward_value = np.array(self.y_reward_value_arr) * self.value_scale

        return np.concatenate([self.x_obs_value, self.x_action_value], axis=1), self.y_reward_value

    def get_value_model(self):
        value_model = Sequential([
            Dense(units=8, input_dim=self.env.observation_space.shape[0] + self.env.action_space.shape[0],
                  activation="relu"),
            Dense(units=4, activation="relu"),
            Dense(units=1, activation="relu")
        ])
        value_model.compile(Adam(learning_rate=value_config["lr"]), "mse")
        return value_model

    def run(self):
        set_memory_growth()

        self.create_value_dataset()
        x, y = self.create_training_data()

