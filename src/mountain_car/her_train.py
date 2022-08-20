import gym
from stable_baselines3 import DDPG, HerReplayBuffer, DQN
from stable_baselines3.her.her import HER

from src.mountain_car.her_env import HerEnv


epsilon = 1-4

env = HerEnv(random_goal=True, epsilon=False)
env_eval = HerEnv(random_goal=False, epsilon=False)

# model = HER("MlpPolicy", env, DDPG, max_episode_length=200, tensorboard_log="tmp/her")
model = DQN('MultiInputPolicy', env, verbose=1, replay_buffer_class=HerReplayBuffer, replay_buffer_kwargs=dict(
        n_sampled_goal=4, goal_selection_strategy="future", max_episode_length=200, online_sampling=True))

model.learn(total_timesteps=3000000, eval_log_path="temp/stable-baselines-ddpg-logs/eval_logs", eval_freq=10000,
            eval_env=env_eval)
