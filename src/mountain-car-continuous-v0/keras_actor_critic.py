from typing import List

import gym
import numpy as np
import tensorflow as tf
from keras.models import load_model
from tensorflow import keras
from tensorflow.keras import layers
from src.utils.gpu import set_memory_growth


set_memory_growth()
run_suffix = "actor_critic_4_no-discount_head-start"

# Configuration parameters for the whole setup
seed = 42
gamma = 1.0  # Discount factor for past rewards
max_steps_per_episode = 200
env = gym.make("MountainCarContinuous-v0")  # Create the environment
env.seed(seed)
eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0


num_inputs = 2
num_actions = 21
actions = np.linspace(-1.0, 1.0, num=num_actions)
num_hidden = 8

# inputs = layers.Input(shape=(num_inputs,))
# common = layers.Dense(num_hidden, activation="relu")(inputs)
# common = layers.Dense(num_hidden, activation="relu")(common)
# action = layers.Dense(num_actions, activation="softmax")(common)
# critic = layers.Dense(1)(common)
#
# model = keras.Model(inputs=inputs, outputs=[action, critic])

optimizer = keras.optimizers.Adam(learning_rate=0.01)
huber_loss = keras.losses.Huber()

model = load_model("temp/keras_actor_critic/model_actor_critic_2_no-discount_head-start.h5")
model.compile(optimizer, huber_loss)

summary_writer = tf.summary.create_file_writer(f"temp/tf-summary_{run_suffix}")


state_history = []
action_probs_history = []
critic_value_history = []
rewards_history = []
running_reward = 0
episode_count = 0

reward_max_vel_weight = 0.0     # 1.0
reward_min_vel_weight = 0.0     # 0.25
reward_max_dist_weight = 1.0    # 1.0
reward_min_dist_weight = 0.0    # 0.125
shaped_reward_scale = 100.0

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


def get_step_diff_shaped_reward(observation) -> float:
    max_vel_reward = np.max([0, observation[1] - max_vel]) * reward_max_vel_weight
    min_vel_reward = np.min([0, observation[1] - min_vel]) * -reward_min_vel_weight
    max_dist_reward = np.max([0, observation[0] - max_dist]) * reward_max_dist_weight
    min_dist_reward = np.min([0, observation[0] - min_dist]) * -reward_min_dist_weight
    shaped_reward = np.max([max_vel_reward, min_vel_reward]) + np.max([max_dist_reward, min_dist_reward])
    return shaped_reward * shaped_reward_scale


frame_count = 0

while True:  # Run until solved
    state = env.reset()
    episode_reward = 0
    with tf.GradientTape() as tape:
        for timestep in range(1, max_steps_per_episode):
            # env.render(); Adding this line would show the attempts
            # of the agent in a pop up window.

            frame_count += 1

            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)

            # Predict action probabilities and estimated future rewards
            # from environment state
            action_probs, critic_value = model(state)
            critic_value_history.append(critic_value[0, 0])

            # Sample action from action probability distribution
            action = np.random.choice(num_actions, p=np.squeeze(action_probs))
            action_probs_history.append(tf.math.log(action_probs[0, action]))

            # Apply the sampled action in our environment
            state, reward, done, _ = env.step([actions[action]])

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

            rewards_history.append(reward)
            state_history.append(state)

            if done:
                break

        # Update running reward to check condition for solving
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

        # Calculate expected value from rewards
        # - At each timestep what was the total reward received after that timestep
        # - Rewards in the past are discounted by multiplying them with gamma
        # - These are the labels for our critic
        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        # Normalize
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()

        # Calculating loss values to update our network
        history = zip(action_probs_history, critic_value_history, returns)
        actor_losses = []
        critic_losses = []
        for log_prob, value, ret in history:
            # At this point in history, the critic estimated that we would get a
            # total reward = `value` in the future. We took an action with log probability
            # of `log_prob` and ended up recieving a total reward = `ret`.
            # The actor must be updated so that it predicts an action that leads to
            # high rewards (compared to critic's estimate) with high probability.
            diff = ret - value
            actor_losses.append(-log_prob * diff)  # actor loss

            # The critic must be updated so that it predicts a better estimate of
            # the future rewards.
            critic_losses.append(
                huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
            )

        # Backpropagation
        loss_value = sum(actor_losses) + sum(critic_losses)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        model.save(f"temp/keras_actor_critic/model_{run_suffix}.h5")

        state_history_chunk = np.array(state_history)
        rewards_history_chunk = np.array(rewards_history)

        pos_std.append(float(np.std(state_history_chunk[:, 0])))
        pos_mean.append(float(np.mean(state_history_chunk[:, 0])))
        speed_std.append(float(np.std(state_history_chunk[:, 1])))
        speed_mean.append(float(np.mean(state_history_chunk[:, 1])))
        reward_std.append(float(np.std(rewards_history_chunk)))
        reward_mean.append(float(np.mean(rewards_history_chunk)))

        with summary_writer.as_default(frame_count):
            tf.summary.scalar("episode_reward", episode_reward)
            tf.summary.scalar("pos_std", pos_std[-1])
            tf.summary.scalar("pos_mean", pos_mean[-1])
            tf.summary.scalar("speed_std", speed_std[-1])
            tf.summary.scalar("speed_mean", speed_mean[-1])
            tf.summary.scalar("reward_std", reward_std[-1])
            tf.summary.scalar("reward_mean", reward_mean[-1])

        # Clear the loss and reward history
        action_probs_history.clear()
        critic_value_history.clear()
        rewards_history.clear()
        state_history.clear()

    # Log details
    episode_count += 1
    if episode_count % 10 == 0:
        template = "running reward: {:.2f} at episode {}"
        print(template.format(running_reward, episode_count))