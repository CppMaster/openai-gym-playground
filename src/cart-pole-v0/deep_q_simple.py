import gym
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, LeakyReLU
from tensorflow import keras


env = gym.make('CartPole-v0')

summary_writer = tf.summary.create_file_writer(f"temp/tf-summary-workshop")

# env parameters
num_actions = env.action_space.n
max_steps_per_episode = 200
# discount factor
gamma = 0.99
# greedy parameter
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.1
epsilon_max = 1.0
epsilon_decline_over_n_episodes = 10000
epsilon_delta = (epsilon_min - epsilon_max) / epsilon_decline_over_n_episodes
# update target network every n episodes
n_episodes_to_update_network = 1000
# train network every en episodes
n_episodes_to_learn = 1

batch_size = 8


# game over punish
end_of_episode_reward = 10.


# a simple model
def get_model():
    return Sequential([
        Dense(units=4, input_dim=env.observation_space.shape[0]),
        LeakyReLU(0.1),
        Dense(units=num_actions, activation="linear")
    ])


# trained model
model = get_model()
# target model
model_target = get_model()
optimizer = keras.optimizers.Adam(learning_rate=0.01, clipnorm=1.0)
loss_function = keras.losses.Huber()

n_episodes_to_score = 10
score_every_n_episodes = 100
render_scoring = False

# experience replay buffer
action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
max_memory_length = 1000

# run infinitely
episode = 0
while True:

    # start the episode
    state = np.array(env.reset())
    episode_reward = 0.0

    # run maximum of 200 steps
    for timestep in range(max_steps_per_episode):

        # render current step
        # env.render()

        # create an input for the model
        state_tensor = tf.convert_to_tensor(state)
        state_tensor = tf.expand_dims(state_tensor, 0)

        # choose a random action
        if np.random.random() > epsilon:
            # inference the model
            action_probs = model(state_tensor, training=False)
            # Take best action
            action = tf.argmax(action_probs[0]).numpy()
        else:
            action = env.action_space.sample()

        # perform the action
        state_next, reward, done, _ = env.step(action)
        state_next = np.array(state_next)

        # accumulate the reward
        episode_reward += reward

        # update replay buffer
        action_history.append(action)
        state_history.append(state)
        state_next_history.append(state_next)
        done_history.append(done)
        rewards_history.append(reward)
        state = state_next

        if done:
            break

    # print(f"Episode finished. Reward: {episode_reward}")
    episode += 1
    with summary_writer.as_default(episode):
        tf.summary.scalar("Training agent reward", episode_reward)

    # train network
    if episode % n_episodes_to_learn == 0 and len(done_history) > batch_size:

        # Get indices of samples for replay buffers
        indices = np.random.choice(range(len(done_history)), size=batch_size)

        # Using list comprehension to sample from replay buffer
        state_sample = np.array([state_history[i] for i in indices])
        state_next_sample = np.array([state_next_history[i] for i in indices])
        rewards_sample = [rewards_history[i] for i in indices]
        action_sample = [action_history[i] for i in indices]
        done_sample = tf.convert_to_tensor([float(done_history[i]) for i in indices])

        # get estimated reward for the next step
        future_rewards = model_target.predict(state_next_sample)
        # apply discounted future rewards
        updated_reward = rewards_sample + gamma * tf.reduce_max(future_rewards, axis=1)
        # if end of episode then no future rewards. The reward is -1 instead, so the agent avoids end of the episode
        updated_reward = updated_reward * (1 - done_sample) - done_sample * end_of_episode_reward

        # create one hot vectors for actions ([1, 0] or [0, 1]), so loss is only calculated for taken action
        masks = tf.one_hot(action_sample, num_actions)

        # track gradient
        with tf.GradientTape() as tape:

            # inference the model
            action_values = model(state_sample)

            # calculate loss only for taken action
            masked_action_value = tf.reduce_sum(tf.multiply(action_values, masks), axis=1)

            loss = loss_function(updated_reward, masked_action_value)
            with summary_writer.as_default(episode):
                tf.summary.scalar("Loss", loss)

        # backpropagation
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # remove old samples from replay experience
    action_history = action_history[-max_memory_length:]
    state_history = state_history[-max_memory_length:]
    state_next_history = state_next_history[-max_memory_length:]
    rewards_history = rewards_history[-max_memory_length:]
    done_history = done_history[-max_memory_length:]

    # update epsilon
    epsilon += epsilon_delta
    epsilon = np.clip(epsilon, epsilon_min, epsilon_max)
    with summary_writer.as_default(episode):
        tf.summary.scalar("Epsilon", epsilon)

    # update target network
    if episode % n_episodes_to_update_network == 0:
        print("Updating target network")
        model_target.set_weights(model.get_weights())

    # score model
    if episode % score_every_n_episodes == 0:
        episode_rewards = []
        for score_episode in range(n_episodes_to_score):
            state = np.array(env.reset())
            episode_reward = 0.0
            for timestep in range(max_steps_per_episode):
                if render_scoring:
                    env.render()
                state_tensor = tf.convert_to_tensor(state)
                state_tensor = tf.expand_dims(state_tensor, 0)
                action_values = model(state_tensor, training=False)
                action = tf.argmax(action_values[0]).numpy()
                state_next, reward, done, _ = env.step(action)
                episode_reward += reward
                state = state_next
                if done:
                    break
            episode_rewards.append(episode_reward)
        mean_reward = np.mean(episode_rewards)
        print(f"Episodes: {episode}. Mean reward: {mean_reward}")
        with summary_writer.as_default(episode):
            tf.summary.scalar("Score agent reward", mean_reward)
