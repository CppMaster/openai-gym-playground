import os.path
from copy import deepcopy
from typing import List, Optional, Tuple, Callable
import logging
import shutil

import gym
import numpy as np
import yaml
from keras.engine.base_layer import Layer
from keras.layers import Dense
from keras.models import Sequential, Model, load_model
from keras.optimizer_v2.adam import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, Callback
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.utils.callbacks.epoch_logger import EpochLogger
from src.utils.dataset import extend_and_save_arr
from src.utils.gpu import set_memory_growth
from src.utils.json_io import save_json, load_json


class MountainCarContinuousTrainer:

    def __init__(self, config_path="config.yml"):

        with open(config_path) as config_file:
            self.config = yaml.load(config_file, Loader=yaml.FullLoader)

        self.file_dir = self.config["run"]["file_dir"]
        os.makedirs(self.file_dir, exist_ok=True)
        shutil.copy(config_path, os.path.join(self.file_dir, "config.yml"))

        self.log = logging.getLogger("")
        self.setup_logs()

        self.env_config = self.config["env"]
        self.env = gym.make(self.env_config["name"])
        self.max_step: int = self.env_config["max_step"]
        self.observation_size: int = self.env.observation_space.shape[0]

        self.value_config = self.config["value"]
        self.n_warmup_sim: int = self.value_config["n_warmup_sim"]
        self.max_pos_episodes: Optional[int] = self.value_config["max_pos_episodes"]
        self.max_pos_episodes: Optional[int] = self.value_config["max_pos_episodes"]
        self.at_most_n_neg_episodes_per_pos: Optional[float] = self.value_config["at_most_n_neg_episodes_per_pos"]
        self.extreme_action_chance: float = self.value_config["extreme_action_chance"]["constant"]
        self.value_scale: float = self.value_config["label_scale"]
        self.min_delta_value: float = self.value_config["min_delta"]
        self.monitor_value: str = self.value_config["monitor"]
        self.x_obs_value_arr: List[List[float]] = []
        self.x_action_value_arr: List[List[float]] = []
        self.y_reward_value_arr: List[List[float]] = []
        self.x_obs_value: np.ndarray = np.empty(0)
        self.x_action_value: np.ndarray = np.empty(0)
        self.y_reward_value: np.ndarray = np.empty(0)
        self.value_model: Optional[Model] = None
        self.value_model_path: str = f"{self.file_dir}/value-model.h5"
        self.value_reward_discount: float = self.value_config["reward"]["discount"]
        self.value_discount_matrix: Optional[np.ndarray] = None
        if self.value_reward_discount:
            self.value_discount_matrix = np.zeros((self.max_step * 2, self.max_step))
            discounts = np.array([np.power(self.value_reward_discount, x) for x in range(self.max_step)])
            for x in range(self.max_step):
                self.value_discount_matrix[x:x + self.max_step, x] = discounts

        self.eval_config = self.config["eval"]
        self.score_n_simulations: int = self.eval_config["n_sim"]
        self.render_every_n_step: Optional[int] = self.eval_config["render_every_n_step"]

        self.policy_config = self.config["policy"]
        self.n_policy_sim: int = self.policy_config["n_sim"]
        self.n_policy_samples: int = self.policy_config["n_policy_samples"]
        self.sample_n_actions: int = self.policy_config["n_action_samples"]
        self.x_obs_policy_arr: List[List[float]] = []
        self.y_action_policy_arr: List[List[float]] = []
        self.x_obs_policy: np.ndarray = np.empty(0)
        self.y_action_policy: np.ndarray = np.empty(0)
        self.min_delta_policy: float = self.policy_config["min_delta"]
        self.monitor_policy: str = self.policy_config["monitor"]
        self.policy_model: Optional[Model] = None
        self.policy_model_path: str = f"{self.file_dir}/policy-model.h5"

    def setup_logs(self):

        log_format = "%(asctime)s :%(levelname)s:\t%(message)s"
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s :%(levelname)s:\t%(message)s",
                            filename=f"{self.file_dir}/log.txt", filemode="a")
        log_formatter = logging.Formatter(log_format)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(log_formatter)
        self.log.addHandler(ch)

    def get_end_shaped_reward(self, observations: List[List[float]]) -> float:
        obs_np = np.array(observations)
        max_vel_reward = (obs_np[:, 1].max() + 0.5) * self.value_config["reward"]["max_vel_weight"]
        min_vel_reward = -(obs_np[:, 1].min() + 0.5) * self.value_config["reward"]["min_vel_weight"]
        max_dist_reward = obs_np[:, 0].max() * self.value_config["reward"]["max_dist_weight"]
        min_dist_reward = -obs_np[:, 0].min() * self.value_config["reward"]["min_dist_weight"]
        shaped_reward = np.max([max_vel_reward, min_vel_reward]) + np.max([max_dist_reward, min_dist_reward])
        return shaped_reward * self.value_config["reward"]["end_reward_scale"]

    def get_step_shaped_reward(self, observation: List[float]) -> float:
        max_vel_reward = (observation[1] + 0.5) * self.value_config["reward"]["max_vel_weight"]
        min_vel_reward = -(observation[1] + 0.5) * self.value_config["reward"]["min_vel_weight"]
        max_dist_reward = observation[0] * self.value_config["reward"]["max_dist_weight"]
        min_dist_reward = observation[0] * self.value_config["reward"]["min_dist_weight"]
        shaped_reward = np.max([max_vel_reward, min_vel_reward]) + np.max([max_dist_reward, min_dist_reward])
        return shaped_reward * self.value_config["reward"]["step_reward_scale"]

    def get_warmup_value_dataset(self) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:

        self.log.info("Creating warmup value dataset")

        x_obs_arr = []
        x_action_arr = []
        y_reward_arr = []

        pos_episodes = 0
        neg_episodes = 0

        def add_episode():
            x_obs_arr.extend(observations)
            x_action_arr.extend(actions)
            y_reward_arr.extend(rewards)

        def do_stop_n_pos_episodes() -> bool:
            return self.max_pos_episodes is not None and pos_episodes >= self.max_pos_episodes

        for _ in tqdm(range(self.n_warmup_sim)):

            if do_stop_n_pos_episodes():
                break

            observation = self.env.reset()
            observations = []
            actions = []
            rewards = []
            is_pos_episode = False

            for _ in range(self.max_step):

                if np.random.random() < self.extreme_action_chance:
                    action = np.array([1.0 if np.random.random() > 0.5 else -1.0])
                else:
                    action = self.env.action_space.sample()
                observations.append(observation)
                actions.append(action)
                observation, reward, done, info = self.env.step(action)
                is_pos_episode = reward > 0.0
                reward += self.get_step_shaped_reward(observation)
                rewards.append(reward)

                if done:
                    break

            if self.value_reward_discount:
                discounted_rewards = deepcopy(rewards)
                n_padding = self.max_step * 2 - len(discounted_rewards)
                discounted_rewards = discounted_rewards + [discounted_rewards[-1]] * n_padding
                discounted_rewards = np.matmul(np.array(discounted_rewards), self.value_discount_matrix)
                rewards = discounted_rewards[:len(rewards)]

            end_shaped_reward = self.get_end_shaped_reward(observations)
            if self.value_config["reward"]["all_steps_have_end_value"]:
                rewards = [np.sum(rewards)] * len(rewards)
            for i in range(len(rewards)):
                rewards[i] += end_shaped_reward

            if is_pos_episode:
                add_episode()
                pos_episodes += 1
                # self.log.debug(f"Successful episode {pos_episodes} with reward: {total_reward}")
                if do_stop_n_pos_episodes():
                    break
            elif self.at_most_n_neg_episodes_per_pos is None or \
                    neg_episodes < self.at_most_n_neg_episodes_per_pos * (pos_episodes + 1):
                add_episode()
                neg_episodes += 1

        self.log.info(f"Successful episodes: {pos_episodes}")

        return x_obs_arr, x_action_arr, y_reward_arr

    def get_incremental_value_dataset(self, step: int = 1) \
            -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:

        self.log.info("Creating incremental value dataset")

        x_obs_arr = []
        x_action_arr = []
        y_reward_arr = []

        n_incremental_sim: int = self.value_config["n_incremental_sim"]
        n_sim_in_batch: int = self.value_config["n_sim_in_batch"]
        n_batches = int(np.ceil(n_incremental_sim / n_sim_in_batch))

        class Simulation:
            def __init__(self, env_name: str, max_step: int, step_shaped_reward_func: Callable[[List[float]], float]):
                self.env = gym.make(env_name)
                self.max_step = max_step
                self.observations = []
                self.actions = []
                self.rewards = []
                self.is_done = False
                self.observation = self.env.reset()
                self.step_shaped_reward_func = step_shaped_reward_func

            def step(self, step_action):
                observation, reward, done, info = self.env.step(step_action)
                reward += self.step_shaped_reward_func(observation)

                self.observations.append(observation)
                self.actions.append(step_action)
                self.rewards.append(reward)
                self.observation = observation
                self.is_done |= done

        exploration_chance: float = self.value_config["exploration"]["constant"]
        exploration_step_factor: Optional[float] = self.value_config["exploration"]["step_factor"]
        exploration_chance = np.power(exploration_step_factor, step) if exploration_step_factor else exploration_chance
        extreme_action_change_step_factor: Optional[float] = self.value_config["extreme_action_chance"]["step_factor"]
        extreme_action_chance = np.power(extreme_action_change_step_factor, step) \
            if extreme_action_change_step_factor else self.extreme_action_chance

        for _ in tqdm(range(n_batches)):
            simulations = [Simulation(self.env_config["name"], self.max_step, self.get_step_shaped_reward)
                           for _ in range(n_sim_in_batch)]

            for _ in range(self.max_step):

                deterministic_sims = [not simulations[i].is_done and np.random.random() > exploration_chance
                                      for i in range(n_sim_in_batch)]

                query_actions = [self.env.action_space.sample() for _ in range(self.sample_n_actions)] + \
                                [np.array([-1.0]), np.array([1.0]), np.array([0.0])]
                n_actions = len(query_actions)

                samples = []
                for i, deterministic_sim in enumerate(deterministic_sims):
                    if not deterministic_sim:
                        continue
                    obs = simulations[i].observation
                    for query_action in query_actions:
                        samples.append(np.concatenate([obs, query_action]))

                x_action = np.array(samples)
                pred = self.value_model.predict(x_action, batch_size=len(samples))
                pred_index = 0
                actions = []
                for i in range(n_sim_in_batch):
                    if deterministic_sims[i]:
                        action = query_actions[np.argmax(
                            pred[pred_index * n_actions: (pred_index + 1) * n_actions], axis=0
                        )[0]]
                        pred_index += 1
                    else:
                        if np.random.random() < extreme_action_chance:
                            action = np.array([1.0 if np.random.random() > 0.5 else -1.0])
                        else:
                            action = self.env.action_space.sample()
                    actions.append(action)
                for i, sim in enumerate(simulations):
                    if not sim.is_done:
                        sim.step(actions[i])
                assert pred_index == sum(deterministic_sims), "Not all pred values were used"

            for i, sim in enumerate(simulations):
                if self.value_reward_discount:
                    discounted_rewards = deepcopy(sim.rewards)
                    n_padding = self.max_step * 2 - len(discounted_rewards)
                    discounted_rewards = discounted_rewards + [discounted_rewards[-1]] * n_padding
                    discounted_rewards = np.matmul(np.array(discounted_rewards), self.value_discount_matrix)
                    sim.rewards = discounted_rewards[:len(sim.rewards)]

                end_shaped_reward = self.get_end_shaped_reward(sim.observations)
                if self.value_config["reward"]["all_steps_have_end_value"]:
                    sim.rewards = [np.sum(sim.rewards)] * len(sim.rewards)
                for j in range(len(sim.rewards)):
                    sim.rewards[j] += end_shaped_reward

                x_obs_arr.extend(sim.observations)
                x_action_arr.extend(sim.actions)
                y_reward_arr.extend(sim.rewards)

        return x_obs_arr, x_action_arr, y_reward_arr

    def create_value_dataset(self, incremental: int = 0):
        self.x_obs_value_arr, self.x_action_value_arr, self.y_reward_value_arr = \
            self.get_incremental_value_dataset(incremental) if incremental else self.get_warmup_value_dataset()
        if self.value_config["load_dataset"]:
            extend_and_save_arr(self.x_obs_value_arr, f"{self.file_dir}/x_obs_arr.p")
            extend_and_save_arr(self.x_action_value_arr, f"{self.file_dir}/x_action_arr.p")
            extend_and_save_arr(self.y_reward_value_arr, f"{self.file_dir}/y_reward_arr.p")

    def create_value_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        self.x_obs_value = np.array(self.x_obs_value_arr)
        self.x_action_value = np.array(self.x_action_value_arr)
        self.y_reward_value = np.array(self.y_reward_value_arr) * self.value_scale

        return np.concatenate([self.x_obs_value, self.x_action_value], axis=1), self.y_reward_value

    def get_value_model(self) -> Model:
        model_config = self.value_config["model"]
        if model_config["load"]:
            if os.path.exists(self.value_model_path):
                return load_model(self.value_model_path)
            else:
                self.log.warning(f"Model file doesn't exists: {self.value_model_path}. Creating a new one")

        layers: List[Layer] = []
        input_dim = self.env.observation_space.shape[0] + self.env.action_space.shape[0]
        for l_i, layer in enumerate(model_config["layers"]):
            if layer["type"] == "dense":
                layers.append(Dense(units=layer["units"], activation=layer["activation"],
                                    input_dim=input_dim if l_i == 0 else None))

        value_model = Sequential(layers)
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
            self.log.debug(f"Score: {total_reward},\tmax distance: {np.max(np.array(observations)[:, 0])}")
            scores.append(total_reward)

        score = float(np.mean(scores))
        self.log.info(f"Avr score: {score}")
        return score

    def train_value_model(self, x: np.ndarray, y: np.ndarray):
        value_callbacks = [
            EpochLogger(),
            ReduceLROnPlateau(patience=10, min_delta=self.min_delta_value, monitor=self.monitor_value, verbose=1),
            EarlyStopping(patience=30, min_delta=self.min_delta_value, monitor=self.monitor_value, verbose=1),
            ScoreModel(os.path.join(self.file_dir, "value-model"), period=self.value_config["score_period"],
                       model=self.value_model, score_func=self.score_value_model)
        ]
        self.log.info("Training value model")

        val_split: Optional[float] = self.value_config["val_split"]
        if val_split:
            x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=val_split, shuffle=False)
        else:
            x_train, x_valid, y_train, y_valid = x, None, y, None

        self.value_model.fit(
            x_train, y_train, validation_data=(x_valid, y_valid), batch_size=self.value_config["batch_size"],
            epochs=self.value_config["epochs"], verbose=0, callbacks=value_callbacks
        )

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
                                + [np.array([-1.0]), np.array([1.0]), np.array([0.0])]
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
                                + [np.array([-1.0]), np.array([1.0]), np.array([0.0])]
                observation = observations[s_i]
                query = [np.concatenate([observation, query_actions[i]]) for i in range(self.sample_n_actions)]
                random_query.extend(query)
            random_query = np.array(random_query)

            pred = self.value_model.predict(random_query, batch_size=self.n_policy_samples * self.sample_n_actions)
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
        self.x_obs_policy_arr, self.y_action_policy_arr = self.get_policy_dataset()
        if self.policy_config["load_dataset"]:
            extend_and_save_arr(self.x_obs_policy_arr, f"{self.file_dir}/xq_obs_arr.p")
            extend_and_save_arr(self.y_action_policy_arr, f"{self.file_dir}/yq_action_arr.p")

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
            EpochLogger(),
            ReduceLROnPlateau(patience=10, min_delta=self.min_delta_policy, monitor=self.monitor_policy, verbose=1),
            EarlyStopping(patience=30, min_delta=self.min_delta_policy, monitor=self.monitor_policy, verbose=1),
            ScoreModel(os.path.join(self.file_dir, "policy-model"), period=self.policy_config["score_period"],
                       model=self.policy_model, score_func=self.score_policy_model)
        ]
        self.log.info("Training policy model")
        self.policy_model.fit(
            x, y, batch_size=self.policy_config["batch_size"],
            epochs=self.policy_config["epochs"], verbose=0, callbacks=policy_callbacks
        )

    def run(self):

        self.log.info("Trainer started")
        self.value_model = self.get_value_model()

        if not self.value_config["skip"]:

            if self.n_warmup_sim:

                self.create_value_dataset(0)
                value_x, value_y = self.create_value_training_data()

                if self.value_config["initial_score"]:
                    self.score_value_model()
                self.train_value_model(value_x, value_y)
                self.score_value_model()

            for incremental_step in range(self.value_config["n_incremental_step"]):
                self.log.info(f"Incremental step: {incremental_step + 1}/{self.value_config['n_incremental_step']}")
                self.create_value_dataset(incremental_step + 1)
                value_x, value_y = self.create_value_training_data()
                if self.value_config["model"]["load"]:
                    self.value_model = self.get_value_model()
                self.train_value_model(value_x, value_y)
                self.score_value_model()

        if not self.policy_config["skip"]:
            self.create_policy_dataset()
            policy_x, policy_y = self.create_policy_training_data()
            self.policy_model = self.get_policy_model()
            self.train_policy_model(policy_x, policy_y)
            self.score_policy_model()


class ScoreModel(Callback):

    def __init__(self, save_path: str, score_func: Callable[[], float], model: Model, period: int = 10):
        super().__init__()
        self.save_path = save_path
        self.model_save_path = f"{self.save_path}.h5"
        self.score_save_path = f"{self.save_path}_score.json"
        self.score_func = score_func
        self.period = period
        self.model = model
        self.best_score = -np.inf
        if os.path.exists(self.score_save_path):
            self.best_score = load_json(self.score_save_path)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.period == 0:
            score = self.score_func()
            if score >= self.best_score:
                self.best_score = score
                self.model.save(f"{self.save_path}.h5")
                save_json(self.best_score, self.score_save_path)


if __name__ == "__main__":
    set_memory_growth()
    mountain_trainer = MountainCarContinuousTrainer()
    mountain_trainer.run()
