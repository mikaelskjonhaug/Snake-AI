from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from SnakeEnv import SnakeEnv 

class RenderCallback(BaseCallback):
    def __init__(self, env, verbose=0):
        super(RenderCallback, self).__init__(verbose)
        self.env = env

    def _on_step(self) -> bool:
        self.env.render()
        return True

training = False
trained = True

if training:
    fps = 1500
    env = SnakeEnv(run_speed=fps) # run_speed = n, n is arbitrary for the # fps the board is being rendered in. 
    model = DQN('MlpPolicy', env, verbose=2) # Deep Q-Network Model, verbose = {0, 1, 2} : live-logging level
    render_callback = RenderCallback(env) # rendering of training
    training_steps = 5E5
    model.learn(total_timesteps=training_steps, callback=render_callback)
    model.save("snake_dqn")


if trained: # Load trained model
    fps = 20
    env = SnakeEnv(run_speed=fps)
    model = DQN.load("snake_dqn")
    obs = env.reset()
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
            obs = env.reset()
