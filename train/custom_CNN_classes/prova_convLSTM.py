from custom_convLSTM import Custom_convLSTM
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models import resnet34
import torch as th
from gym.spaces import Box
import numpy as np
from CNN_env.env_images import custom_hopper
import gym
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.env_checker import check_env

def main():
    model = Custom_convLSTM(observation_space=Box(low=0, high=255, shape=(3, 3, 224, 224), dtype=np.uint8))
    observation_space=Box(low=0, high=255, shape=(3, 3, 224, 224))
    # model = Custom_convLSTM(observation_space=Box(low=0, high=255, shape=(3, 3, 112, 112), dtype=np.uint8))
    model(th.rand(size=(3, 3, 3, 224, 224))) # frames, batch size, channels, H, W
    env = gym.make('CustomHopper-source-v0')
    # check_env(env=env)
    # print(env.reset().shape)
    # print(env.action_space.sample())
    # plt.imshow(env.reset())
    # m = resnet34()
    # train_nodes, eval_nodes = get_graph_node_names(m)
    # print(train_nodes)

if __name__ == '__main__':
    main()
