import gym
import tensorflow as tf
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
import numpy as np
from keras import layers
from tensorflow import keras
from keras.activations import leaky_relu
from keras.regularizers import l1_l2

from src.utils.gpu import set_memory_growth
import matplotlib.pyplot as plt


set_memory_growth()
run_suffix = "max-pooling_dense-bias-regularizer_leaky-relu-01_lose-life-reward-25_min_batch-size-8_lr-0.001"
seed = 42
gamma = 0.99  # Discount factor for past rewards
lose_life_reward = 25
batch_size = 8
min_log_prob = 0.2

env = make_atari("QbertNoFrameskip-v4")
env = wrap_deepmind(env, frame_stack=True, scale=True, clip_rewards=False)
env.seed(seed)

eps = np.finfo(np.float32).eps.item()
num_actions = env.action_space.n


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
action_layer = layers.Dense(num_actions, activation="softmax")(common)
critic = layers.Dense(64, bias_regularizer=l1_l2(l1=0.01, l2=0.01))(common)
critic = layers.LeakyReLU(0.1)(critic)
critic = layers.Dense(1, bias_regularizer=l1_l2(l1=0.01, l2=0.01))(critic)
model = keras.Model(inputs=inputs, outputs=[action_layer, critic])

optimizer = keras.optimizers.Adam(learning_rate=0.001)
huber_loss = keras.losses.Huber()

summary_writer = tf.summary.create_file_writer(f"temp/tf-summary_{run_suffix}")
img_path_dir = f"temp/images"

state_history = []
action_probs_history = []
action_history = []
critic_value_history = []
rewards_history = []
running_reward = 0
episode_count = 0

frame_count = 0

render = False
save_rendered_frames = False

while True:  # Run until solved
    state = env.reset()
    episode_reward = 0
    with tf.GradientTape(persistent=True) as tape:
        while True:
            if render:
                env.render()
            # of the agent in a pop up window.

            frame_count += 1

            state = tf.convert_to_tensor(state)

            if save_rendered_frames:
                fig, axs = plt.subplots(2, 4, clear=True)
                n_frames = state.shape[2]
                for x in range(n_frames):
                    axs[0, x].imshow(state[:, :, x])
                    if x < n_frames - 1:
                        axs[1, x].imshow(np.abs(state[:, :, x + 1] - state[:, :, x]))
                axs[1, n_frames - 1].imshow(np.max(state, axis=2))
                plt.savefig(f"{img_path_dir}/frames_{frame_count}.png")

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
            life_before = env.unwrapped.ale.lives()

            state, reward, done, _ = env.step(action)

            life_after = env.unwrapped.ale.lives()

            episode_reward += reward

            reward += (life_after - life_before) * lose_life_reward
            with summary_writer.as_default(episode_count):
                tf.summary.scalar("Step reward", reward)

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
        for index in range(0, len(actor_losses), batch_size):
            index_slice = slice(index, index + batch_size)
            loss_value = sum(actor_losses[index_slice]) + sum(critic_losses[index_slice])
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
