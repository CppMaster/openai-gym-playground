import gym
import tensorflow as tf
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
import numpy as np
from keras import layers
from tensorflow import keras

from src.utils.gpu import set_memory_growth


set_memory_growth()
run_suffix = "no-frameskip"
seed = 42
gamma = 0.99  # Discount factor for past rewards

env = make_atari("QbertNoFrameskip-v4")
env = wrap_deepmind(env, frame_stack=True, scale=True)
env.seed(seed)

eps = np.finfo(np.float32).eps.item()
num_actions = env.action_space.n

inputs = layers.Input(shape=env.observation_space.shape)
layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)
common = layers.Flatten()(layer3)
action = layers.Dense(num_actions, activation="softmax")(common)
critic = layers.Dense(1)(common)
model = keras.Model(inputs=inputs, outputs=[action, critic])

optimizer = keras.optimizers.Adam(learning_rate=0.01)
huber_loss = keras.losses.Huber()

summary_writer = tf.summary.create_file_writer(f"temp/tf-summary_{run_suffix}")

state_history = []
action_probs_history = []
action_history = []
critic_value_history = []
rewards_history = []
running_reward = 0
episode_count = 0

frame_count = 0

while True:  # Run until solved
    state = env.reset()
    episode_reward = 0
    with tf.GradientTape() as tape:
        while True:
            # env.render()
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
            action_history.append(action)
            action_probs_history.append(tf.math.log(action_probs[0, action]))

            # Apply the sampled action in our environment
            state, reward, done, _ = env.step(action)
            with summary_writer.as_default(episode_count):
                tf.summary.scalar("Step reward", reward)

            episode_reward += reward

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

        model.save_weights(f"temp/keras-actor-critic_{run_suffix}.h5")

        with summary_writer.as_default(episode_count):
            tf.summary.scalar("Total loss", np.mean(actor_losses) + np.mean(critic_losses))
            tf.summary.scalar("Actor loss", np.mean(actor_losses))
            tf.summary.scalar("Critic loss", np.mean(critic_losses))
            tf.summary.histogram("Action", action_history, buckets=num_actions)

        # Clear the loss and reward history
        action_probs_history.clear()
        critic_value_history.clear()
        rewards_history.clear()
        state_history.clear()
        action_history.clear()

    # Log details
    episode_count += 1
    with summary_writer.as_default(episode_count):
        tf.summary.scalar("Episode reward", episode_reward)
        tf.summary.scalar("Running reward", running_reward)

    if episode_count % 10 == 0:
        template = "running reward: {:.2f} at episode {}, last reward {}"
        print(template.format(running_reward, episode_count, episode_reward))
