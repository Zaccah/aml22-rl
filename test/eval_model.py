import gym
from env.custom_hopper import *
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO

def main():
    render = False
    test_env = 'CustomHopper-target-v0'

    env = gym.make(test_env)
    try:
        custom_objects = {
            "lr_schedule": .0003,
            "clip_range": .2
        }
        model = PPO.load(r'trained_models/ppo_standard_source.zip', env=env, custom_objects=custom_objects)
    
    except IOError:
        print("File not found, check the relative path and run in a terminal within the folder")

    # print(model)
    mean_rwd, std_rwd = evaluate_policy(model=model, env=env, n_eval_episodes=50, render=render)
    print(mean_rwd, std_rwd)

if __name__ == '__main__':
    main()