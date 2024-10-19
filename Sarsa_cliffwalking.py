import pickle

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng()

epsilon = 1
learning_rate = 0.9
discount_factor = 0.9
decay_epsilon = 0.0001


def choose_action(state, epsilon):
    if rng.random() < epsilon:
        action = env.action_space.sample()
        return action
    else:
        action = np.argmax(q[state, :])
        return action


env = gym.make("CliffWalking-v0", render_mode="human")
observation, info = env.reset(seed=42)
q = np.zeros((env.observation_space.n, env.action_space.n))
for _ in range(1000):
    observation = env.reset()[0]
    terminated = False
    truncated = False

    while not terminated and not truncated:
        action = choose_action(observation, epsilon)
        new_observation, reward, terminated, truncated, info = env.step(action)
        new_action = choose_action(new_observation, epsilon)
        q[observation, action] = q[observation, action] + learning_rate * (
            reward
            + discount_factor * q[new_observation, new_action]
            - q[observation, action]
        )

        observation = new_observation
    epsilon = max(epsilon - decay_epsilon, 0)


env.close()
