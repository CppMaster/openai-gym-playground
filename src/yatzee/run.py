import numpy as np
import gym
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import MaskablePPO

from src.yatzee.valid_observation_env import YahtzeeValidObservationEnv

env = YahtzeeValidObservationEnv()
monitor_env = Monitor(env)
env = monitor_env

model = MaskablePPO.load("models/vec_env_8")

while True:
    done = False
    obs = env.reset()
    while not done:
        env.render()
        action, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())
        action = action.item()
        obs, reward, done, info = env.step(action)
    print(f"Episode reward: {monitor_env.episode_returns[-1]},\tAverage reward: {np.mean(monitor_env.episode_returns)}")