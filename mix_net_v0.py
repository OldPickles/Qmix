import torch.nn as nn
import torch.nn.functional as F
import torch


class Qmix_network(nn.Module):
    def __init__(self, qmix_hidden_dim=32, hyper_hidden_dim=64, env=None, use_orthogonal=True):
        """

        :param batch: 批量的经验 type: dict
        :param qmix_hidden_dim:
        :param hyper_hidden_dim:
        :param env:
        :param use_orthogonal:
        """
        super(Qmix_network, self).__init__()
        self.env = env
        self.n_agents = env.n_agents

        self.use_orthogonal_init = use_orthogonal
        self.input_dim = self.env.state_dim
        self.qmix_hidden_dim = qmix_hidden_dim
        self.hyper_hidden_dim = hyper_hidden_dim

        self.hyper_w1 = nn.Sequential(nn.Linear(self.input_dim, self.hyper_hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(self.hyper_hidden_dim,
                                                self.n_agents * self.qmix_hidden_dim))
        self.hyper_w2 = nn.Sequential(nn.Linear(self.input_dim, self.hyper_hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(self.hyper_hidden_dim, self.qmix_hidden_dim * 1))

        self.hyper_b1 = nn.Linear(self.input_dim, self.qmix_hidden_dim)
        self.hyper_b2 = nn.Sequential(nn.Linear(self.input_dim, self.qmix_hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(self.qmix_hidden_dim, 1))

    def forward(self, q, state):
        """

        :param q:
            shape = (batch_size, n_agents, self.env.avail_actions_dim)
        :param batch:
            shape = (batch_size, n_agents, info_shape)
            info_shape: 为相关经验的对象形状
        :return:
        """
        q = q.reshape(-1, 1, self.n_agents)
        s = state.reshape(-1, self.input_dim)
        w1 = torch.abs(self.hyper_w1(s))
        b1 = self.hyper_b1(s)
        w1 = w1.view(-1, self.n_agents, self.qmix_hidden_dim)
        b1 = b1.view(-1, 1, self.qmix_hidden_dim)

        q_hidden = F.elu(torch.bmm(q, w1) + b1)

        w2 = torch.abs(self.hyper_w2(s))
        b2 = self.hyper_b2(s)
        w2 = w2.view(-1, self.qmix_hidden_dim, 1)
        b2 = b2.view(-1, 1, 1)

        q_total = torch.bmm(q_hidden, w2) + b2

        return q_total


class VDN_Net(nn.Module):
    def __init__(self, ):
        super(VDN_Net, self).__init__()

    def forward(self, q):
        return torch.sum(q, dim=-1, keepdim=True)  # (batch_size, 1)
