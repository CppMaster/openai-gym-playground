import gym
import math
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as functional


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env = gym.make('MountainCarContinuous-v0')
env.seed(101)
np.random.seed(101)


class Agent(nn.Module):
    def __init__(self, env_, h_size=16):
        super(Agent, self).__init__()
        self.env = env_
        # state, hidden layer, action sizes
        self.s_size = env_.observation_space.shape[0]
        self.h_size = h_size
        self.a_size = env_.action_space.shape[0]
        # define layers (we used 2 layers)
        self.fc1 = nn.Linear(self.s_size, self.h_size)
        self.fc2 = nn.Linear(self.h_size, self.a_size)

    def set_weights(self, weights):
        s_size = self.s_size
        h_size = self.h_size
        a_size = self.a_size
        # separate the weights for each layer
        fc1_end = (s_size * h_size) + h_size
        fc1_w = torch.from_numpy(weights[:s_size*h_size].reshape(s_size, h_size))
        fc1_b = torch.from_numpy(weights[s_size*h_size:fc1_end])
        fc2_w = torch.from_numpy(weights[fc1_end:fc1_end+(h_size*a_size)].reshape(h_size, a_size))
        fc2_b = torch.from_numpy(weights[fc1_end+(h_size*a_size):])
        # set the weights for each layer
        self.fc1.weight.data.copy_(fc1_w.view_as(self.fc1.weight.data))
        self.fc1.bias.data.copy_(fc1_b.view_as(self.fc1.bias.data))
        self.fc2.weight.data.copy_(fc2_w.view_as(self.fc2.weight.data))
        self.fc2.bias.data.copy_(fc2_b.view_as(self.fc2.bias.data))

    def get_weights_dim(self):
        return (self.s_size + 1) * self.h_size + (self.h_size + 1) * self.a_size

    def forward(self, x):
        x = functional.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x.cpu().data

    def act(self, state):
        state = torch.from_numpy(state).float().to(device)
        with torch.no_grad():
            action = self.forward(state)
        return action

    def evaluate(self, weights, gamma=1.0, max_t=5000):
        self.set_weights(weights)
        episode_return = 0.0
        state = self.env.reset()
        for t in range(max_t):
            state = torch.from_numpy(state).float().to(device)
            action = self.forward(state)
            state, reward, done, _ = self.env.step(action)
            episode_return += reward * math.pow(gamma, t)
            if done:
                break
        return episode_return


def cem(agent_, n_iterations=500, max_t=1000, gamma=1.0, print_every=10, pop_size=50, elite_frac=0.2, sigma=0.5):
    """PyTorch's implementation of the cross-entropy method.

    Params
    ======
        Agent (object): agent instance
        n_iterations (int): maximum number of training iterations
        max_t (int): maximum number of timesteps per episode
        gamma (float): discount rate
        print_every (int): how often to print average score (over last 100 episodes)
        pop_size (int): size of population at each iteration
        elite_frac (float): percentage of top performers to use in update
        sigma (float): standard deviation of additive noise
    """
    n_elite = int(pop_size*elite_frac)

    scores_deque = deque(maxlen=100)
    scores_ = []
    # Initialize the weight with random noise
    best_weight = sigma * np.random.randn(agent_.get_weights_dim())

    for i_iteration in range(1, n_iterations+1):
        # Define the candidates and get the reward of each candidate
        weights_pop = [best_weight + (sigma * np.random.randn(agent_.get_weights_dim())) for _ in range(pop_size)]
        rewards = np.array([agent_.evaluate(weights, gamma, max_t) for weights in weights_pop])

        # Select best candidates from collected rewards
        elite_indices = rewards.argsort()[-n_elite:]
        elite_weights = [weights_pop[i] for i in elite_indices]
        best_weight = np.array(elite_weights).mean(axis=0)

        reward = agent_.evaluate(best_weight, gamma=1.0)
        scores_deque.append(reward)
        scores_.append(reward)

        if reward >= np.max(scores_):
            torch.save(agent_.state_dict(), 'checkpoint_3.pth')

        if i_iteration % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}, Last Score: {:.2f}, Best singular reward: {:.2f}'.format(
                i_iteration, np.mean(scores_deque), reward, np.max(rewards))
            )

    return scores_


if __name__ == "__main__":

    print('observation space:', env.observation_space)
    print('action space:', env.action_space)
    print('  - low:', env.action_space.low)
    print('  - high:', env.action_space.high)

    agent = Agent(env).to(device)
    scores = cem(agent, print_every=1, max_t=200, pop_size=1000, sigma=5.0, elite_frac=0.01)
