import gym

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

import numpy as np


env = gym.make('MountainCarContinuous-v0')

model = DDPG('MlpPolicy', env, verbose=1, action_noise=OrnsteinUhlenbeckActionNoise(
    mean=np.zeros(env.action_space.shape), sigma=0.5*np.ones(env.action_space.shape)
), create_eval_env=True, tensorboard_log="stable-baselines-ddpg-logs/tensorboard")
model.learn(total_timesteps=300000, eval_log_path="stable-baselines-ddpg-logs/eval_logs", eval_freq=1000,
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


