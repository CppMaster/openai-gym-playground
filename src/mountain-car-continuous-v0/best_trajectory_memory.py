from typing import Dict, Tuple, Optional, List
from collections import namedtuple
import gym
import numpy as np
import random


MemoryValue = namedtuple("MemoryValue", "action reward")
Trajectory = namedtuple("Trajectory", "state action")


class BestTrajectoryMemoryLearner:

    def __init__(self, env: gym.Env, resolution: int = 1000, exploration_chance: float = 0.5,
                 min_reward_to_record: float = 0.0, n_eval_episodes: int = 5, eval_period: int = 10000,
                 max_steps: int = 200, render_train: bool = False, render_eval: bool = True):
        self.env = env
        self.resolution = resolution
        self.exploration_chance = exploration_chance
        self.min_reward_to_record = min_reward_to_record
        self.memory: Dict[Tuple, MemoryValue] = {}
        self.n_eval_episodes = n_eval_episodes
        self.eval_period = eval_period
        self.max_steps = max_steps
        self.render_train = render_train
        self.render_eval = render_eval

    def learn(self, n_episodes: Optional[int] = None):

        episode = 0
        while n_episodes is None or episode < n_episodes:
            episode_reward = 0.0
            obs = self.env.reset()
            trajectories: List[Trajectory] = []
            for step in range(self.max_steps):
                if self.render_train:
                    self.env.render()
                action = self.policy(observation=obs)
                trajectories.append(Trajectory(obs, action))
                obs, reward, done, info = self.env.step(action)
                episode_reward += reward
                if done:
                    break

            print(f"Episode: {episode}, reward: {episode_reward}")
            if episode_reward >= self.min_reward_to_record:
                self.record_episode(trajectories, episode_reward)

            if len(self.memory) > 0 and episode % self.eval_period == 0:
                scores: List[float] = []
                for _ in range(self.n_eval_episodes):
                    episode_reward = 0.0
                    obs = self.env.reset()
                    for step in range(self.max_steps):
                        if self.render_eval:
                            self.env.render()
                        action = self.policy(observation=obs, deterministic=True)
                        obs, reward, done, info = self.env.step(action)
                        episode_reward += reward
                        if done:
                            break
                    scores.append(episode_reward)
                print(f"Score: {np.mean(scores)}")

            episode += 1

    def policy(self, observation: np.ndarray, deterministic: bool = False) -> List[float]:
        memory_key = self.get_memory_key(observation)
        if memory_key in self.memory:
            memory_value = self.memory[memory_key]
            if deterministic or random.random() > self.exploration_chance:
                return memory_value.action
        # return self.env.action_space.sample()
        return [1.0 if random.random() < 0.5 else -1.0]

    def get_memory_key(self, observation: np.ndarray) -> Tuple:
        return tuple([int(x * self.resolution) for x in observation])

    def record_episode(self, trajectories: List[Trajectory], reward: float):
        for trajectory in trajectories:
            memory_key = self.get_memory_key(trajectory.state)
            if memory_key not in self.memory or self.memory[memory_key].reward <= reward:
                self.memory[memory_key] = MemoryValue(trajectory.action, reward)


if __name__ == '__main__':
    learner = BestTrajectoryMemoryLearner(gym.make('MountainCarContinuous-v0'), resolution=25)
    learner.learn()
