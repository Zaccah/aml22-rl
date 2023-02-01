import gym
from env_images.custom_hopper import *
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecFrameStack

def main():
    render = True
    test_env = 'CustomHopper-source-v0'
    env = gym.make(test_env)
    vect_env = DummyVecEnv([lambda: gym.make('CustomHopper-source-v0')])
    vect_env = VecFrameStack(vect_env, 3)
    # env.render(camera_id=0)
    print(vect_env.observation_space)
    return
    try:
        custom_objects = {
            "lr_schedule": .0003,
            "clip_range": .2
        }
        model = PPO.load(r'trained_models/PPO_vect_env.zip', env=vect_env, custom_objects=custom_objects)
    
    except IOError:
        print("File not found, check the relative path and run in a terminal within the folder")
    
    # print(model)
    mean_rwd, std_rwd = evaluate_policy(model=model, env=env, n_eval_episodes=50, render=render)
    print(mean_rwd, std_rwd)

if __name__ == '__main__':
    main()
    