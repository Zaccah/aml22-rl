import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import resnet34
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

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
            state = (Variable(torch.randn(x.size(0), x.size(1), x.size(2), x.size(3)).cuda()),
                     Variable(torch.randn(x.size(0), x.size(1), x.size(2), x.size(3)).cuda()))
        ht_1, ct_1 = state
        it = torch.sigmoid(self.conv_i_xx(x) + self.conv_i_hh(ht_1))
        ft = torch.sigmoid(self.conv_f_xx(x) + self.conv_f_hh(ht_1))
        ct_tilde = torch.tanh(self.conv_c_xx(x) + self.conv_c_hh(ht_1))
        ct = (ct_tilde * it) + (ct_1 * ft)
        ot = torch.sigmoid(self.conv_o_xx(x) + self.conv_o_hh(ht_1))
        ht = ot * torch.tanh(ct)
        return ht, ct

class ourModel(nn.Module):
    def __init__(self, num_classes=61, mem_size=512, DEVICE="", n_input_channels=3):
        super(ourModel, self).__init__()
        self.DEVICE = DEVICE
        self.num_classes = num_classes
        self.resNet = resnet34(weights='DEFAULT')
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0), # 110, stride 2 pad 1
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0), # 54
            nn.ReLU(),
            nn.Conv2d(64, 512, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )
        self.return_node = {'layer4.2.relu_1': 'feature_layer'}
        self.resnet_feature_extractor = create_feature_extractor(self.resNet, return_nodes=self.return_node)
        self.mem_size = mem_size
        self.weight_softmax = self.resNet.fc.weight
        self.lstm_cell = MyConvLSTMCell(512, mem_size)
        self.avgpool = nn.AvgPool2d(7)
        self.dropout = nn.Dropout(0.7)
        self.fc = nn.Linear(mem_size, self.num_classes)
        self.classifier = nn.Sequential(self.dropout, self.fc)

    def forward(self, inputVariable):

        #Learning with Temporal information (ConvLSTM)
        state = (torch.zeros((inputVariable.size(1), self.mem_size, 7, 7)).to(self.DEVICE),
                torch.zeros((inputVariable.size(1), self.mem_size, 7, 7)).to(self.DEVICE))
        for t in range(inputVariable.size(0)):
            #spatial_frame_feat: (bs, 512, 7, 7)
            # _, spatial_frame_feat, _ = self.resNet(inputVariable[t])
            # spatial_frame_feat = self.cnn(inputVariable[t])
            spatial_frame_feat = self.resnet_feature_extractor(inputVariable[t])
            state = self.lstm_cell(spatial_frame_feat['feature_layer'], state)
        video_level_features = self.avgpool(state[1]).view(state[1].size(0), -1)
        print(video_level_features)
        # avgpool = nn.AvgPool1d(kernel_size=2)
        # print(avgpool(video_level_features))
        logits = self.classifier(video_level_features)
        return logits, video_level_features