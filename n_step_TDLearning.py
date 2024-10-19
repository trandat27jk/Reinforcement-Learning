import random
import gymnasium as gym
import numpy as np

epsilon=1
learning_rate=0.9
discount_factor=0.9
decay_epsilon=0.00001
rng=np.random.default_rng()
n=4 #n_step
env = gym.make(
        "FrozenLake-v1", map_name="8x8", is_slippery=False, render_mode="human"
    )

q=np.zeros((env.observation_space.n,env.action_space.n))

def choose_action(state):
    if rng.random()<epsilon:
        action=env.action_space.sample()
    else:
        action=np.argmax(q[state,:])
    return action
for _ in range(1000):
    state,info=env.reset()
    terminated=False
    truncated=False
    
    states=[state]*n
    actions=[choose_action(state)]*n
    rewards=[0]*n
    t=0
    T=float("inf")
    while t<T:
        if t<T:
            new_state, reward, terminated, truncated, _ = env.step(actions[t%n])
            rewards[t%n]=reward
            states[(t+1)%n]=new_state
            if terminated or truncated:
                T=t+1
            else:
                actions[(t+1)%n]=choose_action(states[(t+1)%n])
                
        
            
        tau=t-n+1
        
        if tau>=0:
            G=0
            for i in range(tau+1,min(tau+n,T)):
                G+=reward[i]
    