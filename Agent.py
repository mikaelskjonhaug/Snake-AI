from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from SnakeEnv import SnakeEnv 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class RenderCallback(BaseCallback):
    def __init__(self, env, verbose=0):
        super(RenderCallback, self).__init__(verbose)
        self.env = env

    def _on_step(self) -> bool:
        self.env.render()
        return True

training = True
trained = False

if training:
    # Train the model
    fps = 1500
    env = Monitor(SnakeEnv(run_speed=fps), filename="./log/") # run_speed = n, n is arbitrary for the # fps the board is being rendered in. 
    model = DQN('MlpPolicy', env, verbose=0) # Deep Q-Network Model, verbose = {0, 1, 2} : live-logging level
    training_steps = 5E4
    render_training = False
    if render_training:
        render_callback = RenderCallback(env) # rendering of training
        model.learn(total_timesteps=training_steps, callback=render_callback)
    else:
        model.learn(total_timesteps=training_steps)
    model.save("snake_dqn")
    env.close()

    # Graph training
    data = pd.read_csv("./log/monitor.csv", skiprows=1)
    rewards = data['r'].values + 1
    episodes = len(data)

    # Average rewards over bin_size # episodes
    # ~100 = episodes / bin_size : to get ~100 data plots.
    bin_size = int(episodes / 100)
    binned_rewards = [np.mean(rewards[i:i+bin_size]) for i in range(0, len(rewards), bin_size)]

    plt.plot(range(len(binned_rewards)), binned_rewards)
    plt.xlabel(f'1 = {bin_size} episodes')
    plt.ylabel('Reward')
    plt.title('Episode vs Reward')
    plt.show()



if trained: # Load trained model
    fps = 10
    env = SnakeEnv(run_speed=fps)
    model = DQN.load("snake_dqn")
    obs, _ = env.reset()
    tot_reward = 0
    steps = 2000
    for _ in range(steps):
        action_arr = model.predict(obs, deterministic=True) #predict() -> action_arr [np arr] : 
        obs, reward, done, info = env.step(int(action_arr[0]))
        tot_reward += reward
        env.render()
        if done:
            print(tot_reward)
            tot_reward = 0
            obs, _ = env.reset()
