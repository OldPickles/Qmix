import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)


class Q_network_RNN(nn.Module):
    def __init__(self, hidden_dim=64, env=None, use_orthogonal=True):
        super(Q_network_RNN, self).__init__()
        self.rnn_hidden = None
        self.env = env

        self.use_orthogonal_init = use_orthogonal

        self.input_dim = self.env.observation_dim
        self.hidden_dim = hidden_dim
        self.output_dim = self.env.avail_actions_dim

        self.rnn_hidden_dim = hidden_dim

        self.fc1 = nn.Linear(self.input_dim, self.rnn_hidden_dim)
        # self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.fc2 = nn.Linear(self.rnn_hidden_dim, self.output_dim)
        if self.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            # orthogonal_init(self.rnn)
            orthogonal_init(self.fc2)

    def forward(self, inputs):
        # When 'choose_action', inputs.shape(N,input_dim)
        # When 'train', inputs.shape(bach_size*N,input_dim)
        inputs = inputs.reshape(-1, self.input_dim)
        x = F.relu(self.fc1(inputs))
        # self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        Q = self.fc2(x)
        return Q

    def action_choice(self, inputs, avail_actions=None, epsilon=0.6):
        """
        如果没有提供avail_actions，则返回Q值最大的动作；
        :param inputs:
        :param avail_actions:
        :param epsilon:
        :return:
        """
        # When 'choose_action', inputs.shape(N,input_dim)
        # When 'train', inputs.shape(bach_size*N,input_dim)
        if avail_actions is None:
            Q = self.forward(inputs)
            return Q.argmax(dim=-1).numpy().reshape((-1, 1))
        else:
            if np.random.uniform() < epsilon:
                return self.env.actions_sample(avail_actions)
            else:
                Q = self.forward(inputs)
                return Q.argmax(dim=-1).cpu().numpy().reshape((-1, 1))



class Q_network_MLP(nn.Module):
    def __init__(self, hidden_dim=64, env=None, use_orthogonal=True):
        super(Q_network_MLP, self).__init__()
        self.rnn_hidden = None
        self.env = env

        self.use_orthogonal_init = use_orthogonal

        self.input_dim = self.env.observation_dim
        self.hidden_dim = hidden_dim
        self.output_dim = self.env.avail_actions_dim

        self.rnn_hidden_dim = hidden_dim

        self.fc1 = nn.Linear(self.input_dim, self.rnn_hidden_dim)
        self.fc2 = nn.Linear(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.fc3 = nn.Linear(self.rnn_hidden_dim, self.action_dim)
        if self.use_orthogonal_init:
            # 判断是否对网络的权重使用正交初始化。
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, inputs, ):
        """
        When 'choose_action',
            inputs.shape(N,input_dim)
            Q.shape = (N, self.env.avail_actions_dim)
        When 'train',
            inputs.shape(bach_size,N,input_dim)
            Q.shape = (batch_size, N, self.env.avail_actions_dim)
        :param inputs:
        :return:
        """
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        Q = self.fc3(x)
        return Q
