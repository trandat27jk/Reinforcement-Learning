import pickle

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

# Initialize random number generator
rng = np.random.default_rng()

# Parameters
epsilon = 1
learning_rate = 0.9
discount_factor = 0.9
decay_epsilon = 0.0001


# policy
def choose_action(state, epsilon):
    if rng.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(q[state, :])
    return action


# expected return value
def expected_return(observation, epsilon):
    policy_probs = np.ones(len(q[observation, :])) * (epsilon / len(q[observation, :]))
    best_action_idx = np.argmax(q[observation, :])
    policy_probs[best_action_idx] += 1 - epsilon
    expected_value = np.sum(policy_probs * q[observation, :])
    return expected_value


env = gym.make("CliffWalking-v0", render_mode="human")


q = np.zeros((env.observation_space.n, env.action_space.n))


for episode in range(1000):
    observation, info = env.reset(seed=42)
    terminated = False
    truncated = False

    while not terminated and not truncated:
        action = choose_action(observation, epsilon)
        new_observation, reward, terminated, truncated, info = env.step(action)

        # Calculate expected value for the next state
        expected_value = expected_return(new_observation, epsilon)

        # Update Q-value using Expected SARSA formula
        q[observation, action] += learning_rate * (
            reward + discount_factor * expected_value - q[observation, action]
        )

        # Update the current observation
        observation = new_observation

    # Decay epsilon after each episode
    epsilon = max(epsilon - decay_epsilon, 0)

env.close()
