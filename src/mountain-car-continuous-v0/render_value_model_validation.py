import gym
from keras.models import load_model
import numpy as np

env = gym.make('MountainCarContinuous-v0')

model = load_model("temp/v12/value-model.h5")

for i_episode in range(20):
    # noinspection PyRedeclaration
    observation = env.reset()
    t = 0
    for t in range(1000):
        env.render()

        x_action = np.concatenate([np.array([observation, observation]), np.array([[-1.0], [1.0]])], axis=1)
        pred = model.predict(x_action, batch_size=2)
        action_index = np.argmax(pred, axis=0)[0]
        action = [-1.0 if action_index == 0 else 1.0]
        print(f"{observation}\t{action}")

        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
