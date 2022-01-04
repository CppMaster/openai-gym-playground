import os.path
from typing import List, Optional, Tuple
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
        self.value_model: Optional[Model] = None

        self.eval_config = self.config["eval"]
        self.score_n_simulations = self.eval_config["n_sim"]
        self.render_every_n_step: Optional[int] = self.eval_config["render_every_n_step"]

        self.policy_config = self.config["policy"]
        self.n_policy_sim = self.policy_config["n_sim"]
        self.n_policy_samples = self.policy_config["n_policy_samples"]
        self.sample_n_actions = self.policy_config["n_action_samples"]
        self.x_obs_policy_arr: List[List[float]] = []
        self.y_action_policy_arr: List[List[float]] = []
        self.x_obs_policy: np.ndarray = np.empty(0)
        self.y_action_policy: np.ndarray = np.empty(0)
        self.min_delta_policy = self.policy_config["min_delta"]
        self.monitor_policy = self.policy_config["monitor"]
        self.policy_model: Optional[Model] = None

    def get_warmup_value_dataset(self) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:

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
            observations = []
            actions = []
            for _ in range(self.max_step):

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

    def create_value_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        self.x_obs_value = np.array(self.x_obs_value_arr)
        self.x_action_value = np.array(self.x_action_value_arr)
        self.y_reward_value = np.array(self.y_reward_value_arr) * self.value_scale

        return np.concatenate([self.x_obs_value, self.x_action_value], axis=1), self.y_reward_value

    def get_value_model(self) -> Model:
        value_model = Sequential([
            Dense(units=8, input_dim=self.env.observation_space.shape[0] + self.env.action_space.shape[0],
                  activation="relu"),
            Dense(units=4, activation="relu"),
            Dense(units=1, activation="relu")
        ])
        value_model.compile(Adam(learning_rate=self.value_config["lr"]), "mse")
        return value_model

    def score_value_model(self) -> float:

        scores = []
        for i_episode in range(self.score_n_simulations):
            observation = self.env.reset()
            total_reward = 0
            observations = []
            for t in range(self.max_step):

                observations.append(observation)

                x_action = np.concatenate([np.array([observation, observation]), np.array([[-1.0], [1.0]])], axis=1)
                pred = self.value_model.predict(x_action, batch_size=2)
                action_index = np.argmax(pred, axis=0)[0]
                action = [-1.0 if action_index == 0 else 1.0]

                if self.render_every_n_step and (t + 1) % self.render_every_n_step == 0:
                    self.env.render()
                observation, reward, done, info = self.env.step(action)
                total_reward += reward
                if done:
                    break
            self.log.debug(f"Score: {total_reward}")
            scores.append(total_reward)

        score = float(np.mean(scores))
        self.log.info(f"Avr score: {score}")
        return score

    def train_value_model(self, x: np.ndarray, y: np.ndarray):
        value_callbacks = [
            ReduceLROnPlateau(patience=10, min_delta=self.min_delta_value, monitor=self.monitor_value, verbose=1),
            EarlyStopping(patience=30, min_delta=self.min_delta_value, monitor=self.monitor_value, verbose=1),
            ScoreValueModel(self, period=self.value_config["score_period"])
        ]
        self.value_model.fit(
            x, y, batch_size=self.value_config["batch_size"], epochs=self.value_config["epochs"], verbose=2,
            callbacks=value_callbacks
        )
        self.value_model.save("temp/car_value-model.h5")

    def get_policy_model(self) -> Model:
        policy_model = Sequential([
            Dense(units=8, input_dim=self.env.observation_space.shape[0], activation="relu"),
            Dense(units=4, activation="relu"),
            Dense(units=self.env.action_space.shape[0], activation="tanh")
        ])
        policy_model.compile(Adam(learning_rate=self.policy_config["lr"]), "mse")
        return policy_model

    def get_policy_dataset(self) -> Tuple[List[List[float]], List[List[float]]]:

        self.log.info("Creating policy dataset")
        xq_obs_arr = []
        yq_action_arr = []

        for _ in tqdm(range(self.n_policy_sim)):
            observation = self.env.reset()
            for t in range(self.max_step):

                query_actions = [self.env.action_space.sample() for _ in range(self.sample_n_actions)] \
                                + [np.array([-1.0]), [np.array([1.0])]]
                query = np.array(
                    [np.concatenate([observation, query_actions[i]]) for i in range(self.sample_n_actions)]
                )
                pred = self.value_model.predict(query, batch_size=self.sample_n_actions)
                best_action = query_actions[np.argmax(pred, axis=0)[0]]

                xq_obs_arr.append(observation)
                yq_action_arr.append(best_action)

                observation, reward, done, info = self.env.step(best_action)

        if self.n_policy_samples:
            observations = [self.env.observation_space.sample() for _ in range(self.n_policy_samples)]
            random_query = []
            for s_i in range(self.n_policy_samples):
                query_actions = [self.env.action_space.sample() for _ in range(self.sample_n_actions)] \
                                + [np.array([-1.0]), [np.array([1.0])]]
                observation = observations[s_i]
                query = [np.concatenate([observation, query_actions[i]]) for i in range(self.sample_n_actions)]
                random_query.extend(query)
            random_query = np.array(random_query)

            pred = self.value_model.predict(random_query, batch_size=self.n_policy_samples*self.sample_n_actions)
            situations = np.split(pred, self.n_policy_samples)
            for s_i, situation in enumerate(situations):
                best_action_index = np.argmax(situation, axis=0)
                query = random_query[best_action_index + s_i * self.sample_n_actions][0]
                best_action = query[self.observation_size:]
                observation = query[:self.observation_size]
                xq_obs_arr.append(observation)
                yq_action_arr.append(best_action)

        return xq_obs_arr, yq_action_arr

    def create_policy_dataset(self):
        os.makedirs("temp", exist_ok=True)
        self.x_obs_policy_arr, self.y_action_policy_arr = self.get_policy_dataset()
        extend_and_save_arr(self.x_obs_policy_arr, "temp/xq_obs_arr.p")
        extend_and_save_arr(self.y_action_policy_arr, "temp/yq_action_arr.p")

    def score_policy_model(self) -> float:

        scores = []
        for i_episode in range(self.score_n_simulations):
            observation = self.env.reset()
            total_reward = 0
            observations = []
            for t in range(self.max_step):

                observations.append(observation)

                action = self.policy_model.predict(np.array([observation]))[0]

                if self.render_every_n_step and (t + 1) % self.render_every_n_step == 0:
                    self.env.render()
                observation, reward, done, info = self.env.step(action)
                total_reward += reward
                if done:
                    break
            self.log.debug(f"Score: {total_reward}")
            scores.append(total_reward)

        score = float(np.mean(scores))
        self.log.info(f"Avr score: {score}")
        return score

    def create_policy_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        self.x_obs_policy = np.array(self.x_obs_policy_arr)
        self.y_action_policy = np.array(self.y_action_policy_arr)
        return self.x_obs_policy, self.y_action_policy

    def train_policy_model(self, x, y):
        policy_callbacks = [
            ReduceLROnPlateau(patience=10, min_delta=self.min_delta_policy, monitor=self.monitor_value, verbose=1),
            EarlyStopping(patience=30, min_delta=self.min_delta_policy, monitor=self.monitor_value, verbose=1),
            ScorePolicyModel(self, period=self.policy_config["score_period"])
        ]
        self.policy_model.fit(
            x, y, batch_size=self.policy_config["batch_size"],
            epochs=self.policy_config["epochs"], verbose=2, callbacks=policy_callbacks
        )
        self.policy_model.save("temp/car_policy-model.h5")

    def run(self):
        set_memory_growth()

        self.create_value_dataset()
        value_x, value_y = self.create_value_training_data()
        self.value_model = self.get_value_model()
        self.train_value_model(value_x, value_y)
        self.score_value_model()

        self.create_policy_dataset()
        policy_x, policy_y = self.create_value_training_data()
        self.policy_model = self.get_policy_model()
        self.train_policy_model(policy_x, policy_y)
        self.score_policy_model()


class ScoreValueModel(Callback):
    def __init__(self, trainer: MountainCarContinuousTrainer, period=10):
        super().__init__()
        self.trainer = trainer
        self.period = period

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.period == 0:
            self.trainer.score_value_model()


class ScorePolicyModel(Callback):
    def __init__(self, trainer: MountainCarContinuousTrainer, period=10):
        super().__init__()
        self.trainer = trainer
        self.period = period

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.period == 0:
            self.trainer.score_policy_model()
