import gym
from environments.env_images_grayscale.custom_hopper import *
from stable_baselines3.common.env_checker import check_env

def main():
    env = gym.make("CustomHopper-source-v0")
    print(env.observation_space)
    print(env.observation_space.sample)
    print(env.reset())
    check_env(env=env)
if __name__ == '__main__':
    main()