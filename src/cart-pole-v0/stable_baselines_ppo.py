import gym
from stable_baselines3 import PPO

env = gym.make("CartPole-v0")

ppo = PPO(env=env, policy="MlpPolicy", n_steps=256, batch_size=256, gae_lambda=0.8, gamma=0.98, n_epochs=20,
          ent_coef=0.0, learning_rate=0.001, clip_range=0.2, tensorboard_log="temp/ppo",
          verbose=True)
ppo.learn(total_timesteps=10000)
