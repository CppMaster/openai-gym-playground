import os.path

from baselines.common.atari_wrappers import MaxAndSkipEnv, EpisodicLifeEnv, WarpFrame, ScaledFloatFrame, FrameStack

from src.utils.gpu import set_memory_growth
import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


set_memory_growth()

run_suffix = "0_ddpg"
save_dir = os.path.join("temp", run_suffix)
os.makedirs(save_dir, exist_ok=True)

seed = 42
skip_frames = 8
stack_frames = 4
reward_scale = 1/25
env_name = "QbertNoFrameskip-v4"
render = True

env = gym.make(env_name)
if skip_frames > 1:
    env = MaxAndSkipEnv(env, skip=skip_frames)
env = EpisodicLifeEnv(env)
env = WarpFrame(env)
env = ScaledFloatFrame(env)
if stack_frames > 1:
    env = FrameStack(env, stack_frames)

env.seed(seed)
num_actions = env.action_space.n
observation_shape = env.observation_space.shape


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
                self.x_prev
                + self.theta * (self.mean - self.x_prev) * self.dt
                + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity,) + observation_shape)
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity,) + observation_shape)

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = np.zeros(shape=(num_actions, ))
        self.action_buffer[index, obs_tuple[1]] = 1.0
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    # @tf.function
    def update(
            self, state_batch, action_batch, reward_batch, next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


def get_actor():
    inputs = layers.Input(shape=env.observation_space.shape)
    layer1 = layers.Conv2D(32, 3)(inputs)
    layer1 = layers.LeakyReLU(0.1)(layer1)
    layer1 = layers.MaxPooling2D((2, 2))(layer1)
    layer2 = layers.Conv2D(64, 3)(layer1)
    layer2 = layers.LeakyReLU(0.1)(layer2)
    layer2 = layers.MaxPooling2D((2, 2))(layer2)
    layer3 = layers.Conv2D(128, 3)(layer2)
    layer3 = layers.LeakyReLU(0.1)(layer3)
    layer3 = layers.MaxPooling2D((2, 2))(layer3)
    common = layers.Flatten()(layer3)
    outputs = layers.Dense(num_actions, activation="softmax")(common)
    model = tf.keras.Model(inputs, outputs)
    return model


def get_critic():
    # State as input
    state_input = layers.Input(shape=env.observation_space.shape)
    layer1 = layers.Conv2D(32, 3)(state_input)
    layer1 = layers.LeakyReLU(0.1)(layer1)
    layer1 = layers.MaxPooling2D((2, 2))(layer1)
    layer2 = layers.Conv2D(64, 3)(layer1)
    layer2 = layers.LeakyReLU(0.1)(layer2)
    layer2 = layers.MaxPooling2D((2, 2))(layer2)
    layer3 = layers.Conv2D(128, 3)(layer2)
    layer3 = layers.LeakyReLU(0.1)(layer3)
    layer3 = layers.MaxPooling2D((2, 2))(layer3)
    state_out = layers.Flatten()(layer3)
    state_out = layers.Dense(32)(state_out)
    state_out = layers.LeakyReLU(0.1)(state_out)

    # Action as input
    action_input = layers.Input(shape=(num_actions,))
    action_out = layers.Dense(32)(action_input)
    action_out = layers.LeakyReLU(0.1)(action_out)

    # Both are passed through separate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(256)(concat)
    out = layers.LeakyReLU(0.1)(out)
    out = layers.Dense(256)(out)
    out = layers.LeakyReLU(0.1)(out)
    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model


"""
`policy()` returns an action sampled from our Actor network plus some noise for
exploration.
"""

min_exploitation_chance = 0.2
max_exploitation_chance = 0.8
chance_range = max_exploitation_chance - min_exploitation_chance
tanh_scale = 0.01

def policy(state, noise_object, mean_reward: float):

    exploitation_chance = min_exploitation_chance + np.tanh(mean_reward * tanh_scale) * chance_range
    if np.random.random() < exploitation_chance:
        sampled_actions = tf.squeeze(actor_model(state))

        # noise = noise_object()
        # sampled_actions = sampled_actions.numpy() + noise

        action = np.argmax(sampled_actions)
    else:
        action = env.action_space.sample()
    return action


"""
## Training hyperparameters
"""

std_dev = 0.2
ou_noise = OUActionNoise(mean=np.zeros(num_actions), std_deviation=float(std_dev) * np.ones(num_actions))

actor_model = get_actor()
critic_model = get_critic()

target_actor = get_actor()
target_critic = get_critic()

# Making the weights equal initially
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

# Learning rate for actor-critic models
critic_lr = 0.002
actor_lr = 0.0001

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

max_step = 200
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.005

buffer = Buffer(50000, 64)

# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []

# Takes about 4 min to train
ep = 0
avg_reward = 0.0
while True:

    prev_state = env.reset()
    episodic_reward = 0

    for timestep in range(max_step):
        # Uncomment this to see the Actor in action
        # But not in a python notebook.
        if render:
            env.render()

        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

        action = policy(tf_prev_state, ou_noise, avg_reward)
        # Recieve state and reward from environment.
        state, reward, done, info = env.step(action)
        reward *= reward_scale
        episodic_reward += reward

        buffer.record((prev_state, action, reward, state))

        buffer.learn()
        update_target(target_actor.variables, actor_model.variables, tau)
        update_target(target_critic.variables, critic_model.variables, tau)

        actor_model.save_weights(f"{save_dir}/actor_{run_suffix}.h5")
        critic_model.save_weights(f"{save_dir}/critic_{run_suffix}.h5")

        target_actor.save_weights(f"{save_dir}/target_actor_{run_suffix}.h5")
        target_critic.save_weights(f"{save_dir}/target_critic_{run_suffix}.h5")

        # End this episode when `done` is True
        if done:
            break

        prev_state = state

    ep_reward_list.append(episodic_reward)

    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-40:])
    print("Episode * {} * Avg Reward is ==> {}, Last Reward: {}".format(ep, avg_reward, ep_reward_list[-1]))
    avg_reward_list.append(avg_reward)

    ep += 1
