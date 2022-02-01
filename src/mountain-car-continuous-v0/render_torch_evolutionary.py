import gym
import torch


from torch_evolutionary import Agent

env = gym.make("MountainCarContinuous-v0")


def show_video_of_model(agent_):
    while True:
        agent_.load_state_dict(torch.load('checkpoint_2.pth'))
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            env.render()

            action = agent_.act(state)
            print(state, action)

            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state
            if done:
                break
        print(total_reward)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
agent = Agent(env).to(device)
show_video_of_model(agent)
