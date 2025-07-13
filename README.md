# Snake-AI

https://github.com/user-attachments/assets/c7126abf-2d52-44aa-8940-d41851fcbbf0

Easily train a reinforcement learning agent to play the classic Snake game using DQN and OpenAI Gym.

## Description

This project uses Deep Q-Learning (DQN) to train an AI agent that learns to play Snake through trial and error. The environment is built with Pygame and wrapped as a Gym-compatible interface for seamless training with Stable-Baselines3. The model will always render as it's learning so to speed up training change the rendering FPS. 


## Dependencies

* Python 3.8+
* pygame
* numpy
* gym
* stable-baselines3

## How to train
### In SnakeEnv.py
* To change the rewards change death penalty / food reward under _calculate_reward()
### In Agent.py
* Set training to True
* Set desired training steps 
* Run Agent.py

## Authors

Mikael Skjonhaug  
[@MikaelSkjonhaug](https://github.com/MikaelSkjonhaug)

## Version History

* 0.1
    * Initial Release

## License

This project is licensed under the MIT License - see the LICENSE.md file for details

## Acknowledgments

* [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
* [OpenAI Gym](https://github.com/openai/gym)
* [Pygame](https://www.pygame.org/)
