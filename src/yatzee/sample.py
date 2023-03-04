import numpy as np
from stable_baselines3.common.monitor import Monitor

from src.yatzee.valid_observation_env import YahtzeeValidObservationEnv

env = YahtzeeValidObservationEnv()
monitor_env = Monitor(env)
env = monitor_env
while True:
    done = False
    env.reset()
    while not done:
        env.render()
        action = env.sample_action()
        observation, reward, done, info = env.step(action)
    print(f"Episode reward: {monitor_env.episode_returns[-1]},\tAverage reward: {np.mean(monitor_env.episode_returns)}")