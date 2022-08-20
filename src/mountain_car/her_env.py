from typing import Dict

import gym
import numpy as np
from gym.envs.classic_control import MountainCarEnv


class HerEnv(gym.GoalEnv):

    def __init__(self, random_goal: bool, epsilon: float = 1-4):
        self.random_goal = random_goal
        self.env: MountainCarEnv = gym.make('MountainCar-v0')
        self.action_space = self.env.action_space

        self.observation_space = gym.spaces.Dict({
            "observation": self.env.observation_space,
            "achieved_goal": gym.spaces.Box(low=self.env.min_position, high=self.env.max_position, shape=(1,)),
            "desired_goal": gym.spaces.Box(low=self.env.min_position, high=self.env.max_position, shape=(1,))
        })
        self.desired_goal = self.get_new_goal()
        self.epsilon = epsilon

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        transformed_obs = self.transform_observation(obs)
        transformed_reward = self.compute_reward(transformed_obs["achieved_goal"], transformed_obs["desired_goal"],
                                                 info)
        return self.transform_observation(obs), transformed_reward, done, info

    def reset(self):
        self.desired_goal = self.get_new_goal()
        obs = self.env.reset()
        return self.transform_observation(obs)

    def get_new_goal(self) -> np.ndarray:
        if self.random_goal:
            return np.random.uniform(self.env.min_position, self.env.max_position, (1,))
        else:
            return np.array([self.env.goal_position])

    def render(self, mode="human"):
        self.env.render(mode)

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info):
        reward = (np.abs(achieved_goal - desired_goal) <= self.epsilon).flatten() * 1.0
        if reward.size == 1:
            reward = reward[0]
        return reward

    def transform_observation(self, obs) -> Dict:
        return {
            "observation": obs,
            "achieved_goal": np.array([obs[0]]),
            "desired_goal": self.desired_goal
        }
