import random

import gymnasium as gym
import numpy as np

# Initialize the environment
env = gym.make("CliffWalking-v0", render_mode="human")
observation, info = env.reset()

# Initialize Q1 and Q2 tables
q1 = np.zeros((env.observation_space.n, env.action_space.n))
q2 = np.zeros((env.observation_space.n, env.action_space.n))

# Hyperparameters
epsilon = 1.0
learning_rate = 0.9
discount_factor = 0.9
decay_epsilon = 0.0001
min_epsilon = 0.01

# Random number generator
rng = np.random.default_rng()


def choose_action(observation, q1, q2, epsilon):
    if rng.random() < epsilon:
        # Exploration: choose a random action
        return env.action_space.sample()
    else:
        # Exploitation: choose action with maximum Q1 + Q2
        total_q = q1[observation, :] + q2[observation, :]
        return np.argmax(total_q)


# Training loop
for episode in range(1000):
    observation, info = env.reset()
    terminated = False
    truncated = False

    while not (terminated or truncated):
        #
        action = choose_action(observation, q1, q2, epsilon)

        new_observation, reward, terminated, truncated, info = env.step(action)

        if random.uniform(0, 1) < 0.5:
            # Update Q1
            q1[observation, action] += learning_rate * (
                reward
                + discount_factor * np.max(q2[new_observation, :])
                - q1[observation, action]
            )
        else:
            # Update Q2
            q2[observation, action] += learning_rate * (
                reward
                + discount_factor * np.max(q1[new_observation, :])
                - q2[observation, action]
            )

        observation = new_observation

    epsilon = max(epsilon - decay_epsilon, min_epsilon)


env.close()
