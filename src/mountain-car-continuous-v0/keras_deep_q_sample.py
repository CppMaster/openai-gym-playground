from typing import List
import gym
import numpy as np
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from tensorflow import keras
from src.utils.gpu import set_memory_growth
import matplotlib.pyplot as plt
import seaborn as sns


set_memory_growth()

run_suffix = 5

# Configuration paramaters for the whole setup
seed = 42
gamma = 0.99  # Discount factor for past rewards
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.1  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
epsilon_interval = (
        epsilon_max - epsilon_min
)  # Rate at which to reduce chance of random action being taken
batch_size = 32  # Size of batch taken from replay buffer
max_steps_per_episode = 200

env = gym.make('MountainCarContinuous-v0')
env.seed(seed)

num_actions = 21
actions = np.linspace(-1.0, 1.0, num=num_actions)

summary_writer = tf.summary.create_file_writer(f"temp/tf-summary_{run_suffix}")

def create_q_model():

    value_model = Sequential([
        Dense(units=8, input_dim=env.observation_space.shape[0], activation="relu"),
        Dense(units=8, activation="relu"),
        Dense(units=num_actions, activation="linear")
    ])

    return value_model


# The first model makes the predictions for Q-values which are used to
# make a action.
model = create_q_model()
# Build a target model for the prediction of future rewards.
# The weights of a target model get updated every 10000 steps thus when the
# loss between the Q-values is calculated the target Q-value is stable.
model_target = create_q_model()

# In the Deepmind paper they use RMSProp however then Adam optimizer
# improves training time
optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
# Using huber loss for stability

loss_function = keras.losses.Huber()
model_target.compile(optimizer, loss_function)

# Experience replay buffers
action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
episode_reward_history = []
running_reward = 0
episode_count = 0
frame_count = 0
# Number of frames to take random action and observe output
epsilon_random_frames = 50000
# Number of frames for exploration
epsilon_greedy_frames = 1000000.0
# Maximum replay length
# Note: The Deepmind paper suggests 1000000 however this causes memory issues
max_memory_length = 100000
# Train the model after 4 actions
update_after_actions = 4
# How often to update the target network
update_target_network = 10000


reward_max_vel_weight = 0.0     # 1.0
reward_min_vel_weight = 0.0     # 0.25
reward_max_dist_weight = 1.0    # 1.0
reward_min_dist_weight = 0.0    # 0.125
shaped_reward_scale = 20.0

max_vel = None
min_vel = None
max_dist = None
min_dist = None

pos_mean: List[float] = []
pos_std: List[float] = []
speed_mean: List[float] = []
speed_std: List[float] = []
reward_mean: List[float] = []
reward_std: List[float] = []


def get_step_shaped_reward(observation) -> float:
    max_vel_reward = (observation[1] + 0.5) * reward_max_vel_weight
    min_vel_reward = -(observation[1] + 0.5) * reward_min_vel_weight
    max_dist_reward = observation[0] * reward_max_dist_weight
    min_dist_reward = observation[0] * reward_min_dist_weight
    shaped_reward = np.max([max_vel_reward, min_vel_reward]) + np.max([max_dist_reward, min_dist_reward])
    return shaped_reward * shaped_reward_scale


def get_step_diff_shaped_reward(observation) -> float:
    max_vel_reward = np.max([0, observation[1] - max_vel]) * reward_max_vel_weight
    min_vel_reward = np.min([0, observation[1] - min_vel]) * -reward_min_vel_weight
    max_dist_reward = np.max([0, observation[0] - max_dist]) * reward_max_dist_weight
    min_dist_reward = np.min([0, observation[0] - min_dist]) * -reward_min_dist_weight
    shaped_reward = np.max([max_vel_reward, min_vel_reward]) + np.max([max_dist_reward, min_dist_reward])
    return shaped_reward * shaped_reward_scale


last_episode_reward = 0

while True:  # Run until solved
    state = np.array(env.reset())
    episode_reward = 0

    max_vel = None
    min_vel = None
    max_dist = None
    min_dist = None

    for timestep in range(1, max_steps_per_episode):
        # env.render(); Adding this line would show the attempts
        # of the agent in a pop up window.
        frame_count += 1

        # Use epsilon-greedy for exploration
        if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
            # Take random action
            action_index = np.random.choice(num_actions)
        else:
            # Predict action Q-values
            # From environment state
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = model(state_tensor, training=False)
            # Take best action
            action_index = tf.argmax(action_probs[0]).numpy()

        # Decay probability of taking random action
        epsilon -= epsilon_interval / epsilon_greedy_frames
        epsilon = max(epsilon, epsilon_min)

        # Apply the sampled action in our environment
        state_next, reward, done, _ = env.step([actions[action_index]])
        state_next = np.array(state_next)

        episode_reward += reward

        if timestep > 1:
            # reward += get_step_shaped_reward(state)
            reward += get_step_diff_shaped_reward(state)

            max_vel = np.max([max_vel, state[1]])
            min_vel = np.min([min_vel, state[1]])
            max_dist = np.max([max_dist, state[0]])
            min_dist = np.min([min_dist, state[0]])
        else:
            max_vel = state[1]
            min_vel = state[1]
            max_dist = state[0]
            min_dist = state[0]

        # Save actions and states in replay buffer
        action_history.append(action_index)
        state_history.append(state)
        state_next_history.append(state_next)
        done_history.append(done)
        rewards_history.append(reward)
        state = state_next

        # Update every fourth frame and once batch size is over 32
        if frame_count % update_after_actions == 0 and len(done_history) > batch_size:

            # Get indices of samples for replay buffers
            indices = np.random.choice(range(len(done_history)), size=batch_size)

            # Using list comprehension to sample from replay buffer
            state_sample = np.array([state_history[i] for i in indices])
            state_next_sample = np.array([state_next_history[i] for i in indices])
            rewards_sample = [rewards_history[i] for i in indices]
            action_sample = [action_history[i] for i in indices]
            done_sample = tf.convert_to_tensor(
                [float(done_history[i]) for i in indices]
            )

            # sns.displot(rewards_history)
            # plt.show()

            # Build the updated Q-values for the sampled future states
            # Use the target model for stability
            future_rewards = model_target.predict(state_next_sample)
            # Q value = reward + discount factor * expected future reward
            updated_q_values = rewards_sample + gamma * tf.reduce_max(
                future_rewards, axis=1
            )

            # If final frame set the last value to -1
            updated_q_values = updated_q_values * (1 - done_sample) - done_sample

            # Create a mask so we only calculate loss on the updated Q-values
            masks = tf.one_hot(action_sample, num_actions)

            with tf.GradientTape() as tape:
                # Train the model on the states and updated Q-values
                q_values = model(state_sample)

                # Apply the masks to the Q-values to get the Q-value for action taken
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                # Calculate loss between new Q-value and old Q-value
                loss = loss_function(updated_q_values, q_action)

            # Backpropagation
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if frame_count % update_target_network == 0:
            # update the the target network with new weights
            model_target.set_weights(model.get_weights())
            model_target.save(f"temp/keras_deep_q_sample/model_{run_suffix}.h5")
            # Log details
            template = "running reward: {:.2f} at episode {}, frame count {}, last episode reward: {:.2f}"
            print(template.format(running_reward, episode_count, frame_count, last_episode_reward))

            state_history_chunk = np.array(state_history[-update_target_network:])
            rewards_history_chunk = np.array(rewards_history[-update_target_network:])

            pos_std.append(float(np.std(state_history_chunk[:, 0])))
            pos_mean.append(float(np.mean(state_history_chunk[:, 0])))
            speed_std.append(float(np.std(state_history_chunk[:, 1])))
            speed_mean.append(float(np.mean(state_history_chunk[:, 1])))
            reward_std.append(float(np.std(rewards_history_chunk)))
            reward_mean.append(float(np.mean(rewards_history_chunk)))

            with summary_writer.as_default(frame_count):
                tf.summary.scalar("pos_std", pos_std[-1])
                tf.summary.scalar("pos_mean", pos_mean[-1])
                tf.summary.scalar("speed_std", speed_std[-1])
                tf.summary.scalar("speed_mean", speed_mean[-1])
                tf.summary.scalar("reward_std", reward_std[-1])
                tf.summary.scalar("reward_mean", reward_mean[-1])

            # fig = plt.figure()
            # plt.subplot(2, 3, 1)
            # plt.title("pos_std")
            # plt.plot(pos_std, label="pos_std")
            # plt.subplot(2, 3, 4)
            # plt.title("pos_mean")
            # plt.plot(pos_mean, label="pos_mean")
            # plt.subplot(2, 3, 2)
            # plt.title("speed_std")
            # plt.plot(speed_std, label="speed_std")
            # plt.subplot(2, 3, 5)
            # plt.title("speed_mean")
            # plt.plot(speed_mean, label="speed_mean")
            # plt.subplot(2, 3, 3)
            # plt.title("reward_std")
            # plt.plot(reward_std, label="reward_std")
            # plt.subplot(2, 3, 6)
            # plt.title("reward_mean")
            # plt.plot(reward_mean, label="reward_mean")
            # plt.close(fig)
            # plt.savefig("plot.png")



        # Limit the state and reward history
        if len(rewards_history) > max_memory_length:
            del rewards_history[:1]
            del state_history[:1]
            del state_next_history[:1]
            del action_history[:1]
            del done_history[:1]

        if done:
            break

    with summary_writer.as_default(frame_count):
        tf.summary.scalar("episode_reward", episode_reward)

    last_episode_reward = episode_reward

    # Update running reward to check condition for solving
    episode_reward_history.append(episode_reward)
    if len(episode_reward_history) > 100:
        del episode_reward_history[:1]
    running_reward = np.mean(episode_reward_history)

    episode_count += 1