from stable_baselines3 import DQN

from src.mountain_car.her_env import HerEnv


env_eval = HerEnv(random_goal=False, epsilon=1e-4)
model = DQN.load("temp/her/DQN_4/model.zip", env=env_eval)


while True:
    total_rewards = 0.0
    obs = env_eval.reset()
    done = False
    while not done:
        env_eval.render()
        action = model.predict(obs)[0]
        obs, reward, done, info = env_eval.step(action)
        total_rewards += reward
    print(f"Total rewards: {total_rewards}")
