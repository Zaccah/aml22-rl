import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import resnet34
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision.models.feature_extraction import create_feature_extractor
from gym import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

DEVICE = 'cuda:0'

class MyConvLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, kernel_size=3, stride=1, padding=1):
        super(MyConvLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv_i_xx = nn.Conv2d(input_size, hidden_size, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_i_hh = nn.Conv2d(hidden_size, hidden_size, kernel_size=kernel_size, stride=stride, padding=padding,
                                   bias=False)

        self.conv_f_xx = nn.Conv2d(input_size, hidden_size, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_f_hh = nn.Conv2d(hidden_size, hidden_size, kernel_size=kernel_size, stride=stride, padding=padding,
                                   bias=False)

        self.conv_c_xx = nn.Conv2d(input_size, hidden_size, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_c_hh = nn.Conv2d(hidden_size, hidden_size, kernel_size=kernel_size, stride=stride, padding=padding,
                                   bias=False)

        self.conv_o_xx = nn.Conv2d(input_size, hidden_size, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_o_hh = nn.Conv2d(hidden_size, hidden_size, kernel_size=kernel_size, stride=stride, padding=padding,
                                   bias=False)

        torch.nn.init.xavier_normal_(self.conv_i_xx.weight)
        torch.nn.init.constant_(self.conv_i_xx.bias, 0)
        torch.nn.init.xavier_normal_(self.conv_i_hh.weight)

        torch.nn.init.xavier_normal_(self.conv_f_xx.weight)
        torch.nn.init.constant_(self.conv_f_xx.bias, 0)
        torch.nn.init.xavier_normal_(self.conv_f_hh.weight)

        torch.nn.init.xavier_normal_(self.conv_c_xx.weight)
        torch.nn.init.constant_(self.conv_c_xx.bias, 0)
        torch.nn.init.xavier_normal_(self.conv_c_hh.weight)

        torch.nn.init.xavier_normal_(self.conv_o_xx.weight)
        torch.nn.init.constant_(self.conv_o_xx.bias, 0)
        torch.nn.init.xavier_normal_(self.conv_o_hh.weight)

    def forward(self, x, state):
        if state is None:
            state = (Variable(torch.randn(x.size(0), x.size(1), x.size(2), x.size(3))),
                     Variable(torch.randn(x.size(0), x.size(1), x.size(2), x.size(3))))
        ht_1, ct_1 = state
        it = torch.sigmoid(self.conv_i_xx(x) + self.conv_i_hh(ht_1))
        ft = torch.sigmoid(self.conv_f_xx(x) + self.conv_f_hh(ht_1))
        ct_tilde = torch.tanh(self.conv_c_xx(x) + self.conv_c_hh(ht_1))
        ct = (ct_tilde * it) + (ct_1 * ft)
        ot = torch.sigmoid(self.conv_o_xx(x) + self.conv_o_hh(ht_1))
        ht = ot * torch.tanh(ct)
        return ht, ct

class Custom_convLSTM(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        # self.DEVICE = DEVICE
        self.resNet = resnet34(weights='DEFAULT')
        self.return_node = {'layer4.2.relu_1': 'feature_layer'}
        self.resnet_feature_extractor = create_feature_extractor(self.resNet, return_nodes=self.return_node)
        self.mem_size = 512
        self.lstm_cell = MyConvLSTMCell(512, 512)
        self.avgpool = nn.AvgPool2d(7)
        self.lstm_cell.train(True)
        self.resNet.train(False)
        # n_input_channels = 3
        # self.cnn = nn.Sequential(
        #     nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
        #     nn.ReLU(),
        #     # nn.Flatten(),
        # )

        # Compute shape by doing one forward pass
        # with torch.no_grad():
        #     cnn_output = self.cnn(torch.as_tensor(observation_space.sample()[None]).float().squeeze())
        #     n_flatten = cnn_output.shape[1]

        # self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

        with torch.no_grad():
            state = (torch.zeros((3, self.mem_size, 7, 7)),
                torch.zeros((3, self.mem_size, 7, 7)))
            spatial_frame_feat = self.resnet_feature_extractor(torch.rand(size=(3, 3, 224, 224)))
            state = self.lstm_cell(spatial_frame_feat['feature_layer'], state)
            movement_features = self.avgpool(state[1]).view(state[1].size(0), -1)
            n_flatten = movement_features.shape[1]
            # n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
        

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        state = (torch.zeros((observations.size(1), self.mem_size, 7, 7)).to(DEVICE),
                torch.zeros((observations.size(1), self.mem_size, 7, 7)).to(DEVICE))
        for t in range(observations.size(0)):
            #spatial_frame_feat: (bs, 512, 7, 7)
            # _, spatial_frame_feat, _ = self.resNet(inputVariable[t])
            # spatial_frame_feat = self.cnn(inputVariable[t])
            spatial_frame_feat = self.resnet_feature_extractor(observations[t])
            state = self.lstm_cell(spatial_frame_feat['feature_layer'].to(DEVICE), state)
        movement_features = self.avgpool(state[1]).view(state[1].size(0), -1)
        # print(self.linear(torch.mean(movement_features, dim=0)).shape)
        return_value = self.linear(torch.mean(movement_features, dim=0))
        return self.linear(torch.mean(movement_features, dim=0))