import gym
from keras.models import load_model
from gym.wrappers.monitoring import video_recorder
import numpy as np


env = gym.make('CartPole-v0')
vid = video_recorder.VideoRecorder(env, path="temp/{}.mp4".format('CartPole-v0'))
model = load_model(r"temp/cartpole_to_video.h5")
state = env.reset()
done = False
while not done:
    vid.capture_frame()

    x_action = np.concatenate([np.array([state, state]), np.array([[0.0], [1.0]])], axis=1)
    pred = model.predict(x_action, batch_size=2)
    action = np.argmax(pred, axis=0)[0]

    next_state, reward, done, _ = env.step(action)
    state = next_state
    if done:
        break

env.close()

