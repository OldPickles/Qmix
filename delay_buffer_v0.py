import numpy as np


class DelayBuffer:
    def __init__(self, batch_size=32, buffer_capacity=5000, env=None):
        """
        :param env:
        """
        self.env = env
        self.n_agents = env.n_agents
        self.buffer_capacity = buffer_capacity
        self.memory_size = 0
        self.memory_index = 0
        self.batch_size = batch_size
        # 经验池实体
        self.buffer = {
            "last_states": np.zeros(((self.buffer_capacity, self.n_agents,) + env.state_shape)),
            "next_states": np.zeros(((self.buffer_capacity, self.n_agents,) + env.state_shape)),
            "last_observations": np.zeros(((self.buffer_capacity, self.n_agents,) + env.observation_shape)),
            "next_observations": np.zeros(((self.buffer_capacity, self.n_agents,) + env.observation_shape)),
            "actions": np.zeros(((self.buffer_capacity, self.n_agents,) + env.action_shape), dtype=np.int32),
            "avail_actions": np.zeros(((self.buffer_capacity, self.n_agents,) + env.avail_actions_shape)),
            "next_avail_actions": np.zeros(((self.buffer_capacity, self.n_agents,) + env.avail_actions_shape)),
            "rewards": np.zeros((self.buffer_capacity, self.n_agents,) + self.env.reward_shape),
            "dones": np.zeros((self.buffer_capacity, self.n_agents,) + self.env.done_shape),
            "terminate": np.zeros((self.buffer_capacity, 1, 1))
        }

    def store_transition(self, last_states, next_states, last_observations, next_observations, actions,
                         avail_actions, next_avail_actions, rewards, dones, terminate):
        """
        存储经验
        """
        self.memory_size = min(self.memory_size + 1, self.buffer_capacity)
        for key in self.buffer.keys():
            # eval,通过变量的内容获取变量
            self.buffer[key][self.memory_index] = eval(key)
        self.memory_index = (self.memory_index + 1) % self.buffer_capacity

    def sample(self):
        """
        随机抽取batch_size个经验
        """
        index = np.random.choice(self.memory_size, size=self.batch_size, replace=False)
        batch = {}
        for key in self.buffer.keys():
            batch[key] = self.buffer[key][index]
        return batch
