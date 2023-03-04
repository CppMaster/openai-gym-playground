import gym
from gym import ObservationWrapper
from gym.spaces import Dict

from stable_baselines3 import DDPG, PPO, HerReplayBuffer, SAC
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

import numpy as np
from stable_baselines3.her.her import HER

env = gym.make('MountainCarContinuous-v0')


# class DictObservationWrapper(ObservationWrapper):
#
#     def __init__(self, env: gym.Env):
#         super().__init__(env)
#         self._observation_space = Dict({"obs": env.observation_space})
#
#     def observation(self, observation):
#         return {"obs": observation}
#
#
# env = DictObservationWrapper(env)

# model = PPO('MlpPolicy', env, verbose=1, policy_kwargs=dict(ortho_init=True),
#             create_eval_env=True, tensorboard_log="temp/stable-baselines-ddpg-logs/tensorboard")
# model = SAC('MultiInputPolicy', env, verbose=1, replay_buffer_class=HerReplayBuffer, replay_buffer_kwargs=dict(
#         n_sampled_goal=4, goal_selection_strategy="future", max_episode_length=100, online_sampling=True)
#              )
model = HER("MlpPolicy", env, DDPG, max_episode_length=100)

model.learn(total_timesteps=3000000, eval_log_path="temp/stable-baselines-ddpg-logs/eval_logs", eval_freq=1000,
            eval_env=gym.make('MountainCarContinuous-v0'))

for _ in range(10):
    obs = env.reset()
    total_reward = 0
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        env.render()
        if done:
            print("Total reward:", total_reward)
            break


