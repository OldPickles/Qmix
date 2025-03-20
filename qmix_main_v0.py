import time
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from agent_flag_v1 import Environment
from delay_buffer_v0 import DelayBuffer

from Agent_net import Q_network_RNN
from mix_net_v0 import Qmix_network


class QMIX_algo:
    def __init__(self, env=None, epochs=50000, lr=0.0001, gamma=0.7, epsilon=0.9,
                 epsilon_decay_step=2000, epsilon_min=0.1, buffer_size=50, batch_size=32,
                 target_update_interval=200, model_version="v*", epoch_print_interval=1, seed=43):
        # 环境初始化
        self.env = env
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.epochs = epochs
        self.epoch_print_interval = epoch_print_interval
        self.seed = seed
        self.model_version = model_version

        # 训练参数
        self.model_save_path = os.path.join('models', self.model_version)

        # 模型超参
        self.epsilon = epsilon
        self.epsilon_decay_step = epsilon_decay_step
        self.epsilon_min = epsilon_min
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.epsilon_decay_step  # epsilon一次衰减值
        self.lr = lr
        self.target_update_interval = target_update_interval
        self.gamma = gamma
        self.max_steps = self.env.WIDTH * self.env.HEIGHT

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 定义网络
        self.eval_agent_net = Q_network_RNN(env=env).to(self.device)
        self.target_agent_net = Q_network_RNN(env=env).to(self.device)
        self.eval_qmix_net = Qmix_network(env=env).to(self.device)
        self.target_qmix_net = Qmix_network(env=env).to(self.device)

        self.target_agent_net.load_state_dict(self.eval_agent_net.state_dict())
        self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())

        # 优化器, 损失函数
        self.eval_agent_optimizer = torch.optim.Adam(self.eval_agent_net.parameters(), lr=self.lr)
        self.eval_qmix_optimizer = torch.optim.Adam(self.eval_qmix_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()
        self.delay_buffer = DelayBuffer(env=env, batch_size=self.batch_size, buffer_capacity=self.buffer_size)

    def update_target_net(self):
        """
        更新目标网络
        :return:
        """
        self.target_agent_net.load_state_dict(self.eval_agent_net.state_dict())
        self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())

    def run(self):
        """
        算法运行主体
        :return:
        """
        train_times = 0
        evaluate_index = {
            "reward_sum": [],
            "flag_num_per_step": [],
            "steps_per_episode": [],
            "flag_num": []
        }

        # 打印指标
        n_success = 0  # 成功找到所有旗子的回合数
        n_find_flag = 0  # 找到旗子的回合数

        # 训练开始
        print(f"训练开始! model_version: {self.model_version}")
        for epoch in range(self.epochs):
            self.env.reset()

            # 绘图评价指标
            rewards_sum = 0
            flag_num_per_step = 0
            steps_per_episode = 0
            flag_num = 0

            # epsilon 衰减
            self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)

            for step in range(self.max_steps):
                # time.sleep(0)
                last_observations = self.env.get_observations()
                last_states = self.env.get_state()
                avail_actions = self.env.get_avail_actions()
                actions = self.choose_action(last_observations, avail_actions)

                dones, rewards, info = self.env.step(actions)
                next_avail_actions = self.env.get_avail_actions()
                # 如果已经结束了，那么再求next_observations会报错，所以这里需要判断一下
                next_observations = self.env.get_observations()
                next_states = self.env.get_state()
                terminate = 1 if self.env.is_done() else 0
                self.delay_buffer.store_transition(last_states, next_states, last_observations, next_observations,
                                                   actions, avail_actions, next_avail_actions, rewards, dones,
                                                   terminate)

                # 添加相关评价指标
                rewards_sum += int(rewards.sum())
                if self.delay_buffer.memory_size >= self.batch_size:
                    batch = self.delay_buffer.sample()
                    self.train(batch)
                    train_times += 1

                # 如果达到更新间隔, 更新目标网络
                if train_times+1 % self.target_update_interval == 0:
                    self.update_target_net()

                # 如果回合结束
                if any(dones) or step == self.max_steps-1:
                    # 更新评价指标
                    flag_num_per_step += (self.env.n_flag - len(self.env.flags)) / (step+1)
                    steps_per_episode += step+1
                    flag_num += self.env.n_flag - len(self.env.flags)
                    evaluate_index["reward_sum"].append(rewards_sum)
                    evaluate_index["flag_num_per_step"].append(flag_num_per_step)
                    evaluate_index["steps_per_episode"].append(steps_per_episode)
                    evaluate_index["flag_num"].append(flag_num)

                    # 更新打印指标
                    if self.env.is_done():
                        n_success += 1  # 成功找到所有旗子的回合数
                    if self.env.n_flag - len(self.env.flags) > 0:
                        n_find_flag += 1  # 找到旗子的回合数

                    # 额外信息

                    if epoch % self.epoch_print_interval == 0:
                        # 每epoch_print_interval 次 epoch打印一次训练信息
                        print(f'Epoch: {epoch}\tSteps: {step+1}')
                    break
        # 输出评价指标曲线
        self.evaluate_policy(evaluate_index)

        # 输出打印指标
        print(f"成功找到所有旗子的回合比率:{n_success/self.epochs:.2f}")
        print(f"找到旗子的回合比率:{n_find_flag/self.epochs:.2f}")

        # 保存信息
        with open(os.path.join(self.model_save_path, "evaluate_index.txt"), "w") as f:
            f.write(f"成功找到所有旗子的回合比率:{n_success/self.epochs:.2f}\n")
            f.write(f"找到旗子的回合比率:{n_find_flag/self.epochs:.2f}")

        # 训练结束后测试模型
        self.test()

        # 训练结束, 收尾工作
        self.end()

    def end(self):
        """
        收尾工作
        工作1: 保存模型
        工作2: 关闭环境
        :return:
        """
        self.env.destroy()
        print(f"训练结束!")

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        torch.save(self.target_agent_net.state_dict(), os.path.join(self.model_save_path, "target_agent_net.pth"))
        torch.save(self.target_qmix_net.state_dict(), os.path.join(self.model_save_path, "target_qmix_net.pth"))
        print(f"模型保存成功!")

    def train(self, batch):
        """
        根据给出的经验:batch 训练模型
        :return:
        """
        # agent_net
        q_net_eval_input = torch.tensor(batch["last_observations"], requires_grad=True).float().to(self.device)
        q_values_eval = self.eval_agent_net.forward(q_net_eval_input).reshape((-1, self.env.n_workers,) +
                                                                              self.env.avail_actions_shape)
        q_net_target_input = torch.tensor(batch["next_observations"], requires_grad=True).float().to(self.device)
        q_values_target = self.target_agent_net.forward(q_net_target_input).reshape((-1, self.env.n_workers,) +
                                                                                    self.env.avail_actions_shape)
        # 将q_values_eval中的q值,依据所选取的动作取出来对应q值,
        # shape: batch_size * a_agents * avail_actions_shape -> batch_size * a_agents
        actions = torch.tensor(batch["actions"], dtype=torch.int64).to(self.device)
        q_values_eval = q_values_eval.gather(dim=-1, index=actions)
        q_values_eval = q_values_eval.squeeze()  # shape = (batch_size, a_agents)
        pass
        # 将q_values_target中的q值,依据下一步可执行的动作,选取其中最大的q值.
        # shape: batch_size * a_agents * avail_actions_shape -> batch_size * a_agents
        next_avail_actions = torch.tensor(batch["next_avail_actions"], dtype=torch.int64).to(self.device)
        q_values_target[next_avail_actions == 0] = -np.inf
        q_values_target = q_values_target.max(dim=-1, keepdim=True)[0].squeeze()  # shape = batch_size * a_agents
        pass

        # qmix_net
        mix_net_eval_input = torch.tensor(batch['last_states'], requires_grad=True).float().to(self.device)
        mix_net_eval_input = mix_net_eval_input[:, 0::self.env.n_workers, :]  # 在一个批次里面,所有智能体的全局状态都是一样的.
        mix_net_target_input = torch.tensor(batch['next_states'], requires_grad=True).float().to(self.device)
        mix_net_target_input = mix_net_target_input[:, 0::self.env.n_workers, :]  # 在一个批次里面,所有智能体的全局状态都是一样的.

        q_total_eval = self.eval_qmix_net.forward(q_values_eval, mix_net_eval_input)
        q_total_eval = q_total_eval.squeeze()  # shape = (batch_size,)
        q_total_target = self.target_qmix_net.forward(q_values_target, mix_net_target_input)
        q_total_target = q_total_target.squeeze()  # shape = (batch_size,)

        # 计算奖励
        rewards = torch.tensor(batch['rewards']).float().to(self.device)
        rewards = rewards.squeeze()  # 使得rewards的shape为 (batch_size, n_agents)
        rewards = torch.sum(rewards, dim=-1, keepdim=True)
        rewards = rewards.squeeze()  # 使得rewards的shape为 (batch_size,)

        # 回合是否结束
        terminates = torch.tensor(batch['terminate'],).float().to(self.device)
        terminates = terminates.squeeze()  # 使得terminates的shape为 (batch_size,)

        # 目标q值
        q_total_target = rewards + self.gamma * (1 - terminates) * q_total_target

        # 损失函数
        # torch.nn.MSELoss()  # 均方误差
        # loss = self.loss_func(q_total_eval, q_total_target)
        # 时序误差
        loss = (q_total_eval - q_total_target.detach()) ** 2

        # 梯度清零
        self.eval_agent_optimizer.zero_grad()
        self.eval_qmix_optimizer.zero_grad()

        # 反向传播
        loss.sum().backward()
        self.eval_agent_optimizer.step()
        self.eval_qmix_optimizer.step()

    def test(self):
        """
        测试模型: 执行10次网络确定的policy self.epsilon = 0.0
        画出找到旗子的数量折线图
        :return:
        """
        print("this is a test function.")

        test_epochs = 10
        evaluate_dict = {
            'flag_num':[],
        }
        for _ in range(test_epochs):
            self.env.reset()
            flag_num = 0
            for step in range(self.max_steps):
                last_observations = self.env.get_observations()
                last_states = self.env.get_state()
                avail_actions = self.env.get_avail_actions()
                actions = self.choose_action(last_observations, avail_actions, use_epsilon=False)
                dones, rewards, info = self.env.step(actions)
                next_avail_actions = self.env.get_avail_actions()
                next_observations = self.env.get_observations()
                next_states = self.env.get_state()
                terminate = 1 if self.env.is_done() else 0
                self.delay_buffer.store_transition(last_states, next_states, last_observations, next_observations,
                                                   actions, avail_actions, next_avail_actions, rewards, dones,
                                                   terminate)

                # 旗子数量
                flag_num += self.env.n_flag - len(self.env.flags)


                if any(dones) or step == self.max_steps-1:
                    evaluate_dict['flag_num'].append(flag_num)
                    break
        plt.rcParams['font.size'] = 10
        plt.figure(figsize=(10, 8))

        plt.plot(list(range(len(evaluate_dict['flag_num']))), evaluate_dict['flag_num'])
        plt.xlabel("Epoch")
        plt.ylabel("Flag Num")
        plt.title("Test Result")
        plt.savefig(os.path.join(self.model_save_path, "test_result.png"))
        plt.show()
        plt.close()

    def choose_action(self, observations, avail_actions, use_epsilon=True):
        """
        根据给定的状态和可用的动作选择动作
        :param observations:
        :param avail_actions:
        :param use_epsilon:
        :return:
        """
        if use_epsilon:
            self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)
        else:
            self.epsilon = 0.0
        if observations.device != self.device:
            observations = observations.to(self.device)
        return self.eval_agent_net.action_choice(observations, avail_actions, self.epsilon)

    def evaluate_policy(self, evaluate_index_dict):
        """
        评估指标
        横坐标为: range(self.epochs)
            指标1: 总获取的rewards之和
            指标2: 平均每步找到多少旗子, 可以是小数.
            指标3: 完成任务的步数
            指标4: 找到旗子的总数
        :param:
            rewards_sum type:list
            flag_num_per_step type:list
            steps_per_episode type:list
            flag_num type:list
        :return:
        """
        fig_number = len(evaluate_index_dict)
        plt.rcParams['font.size'] = 140
        fig, axes = plt.subplots(1, fig_number, figsize=(100, 30))
        line_styles = ['g-', 'r-', 'b-', 'y-']
        for i, (key, value) in enumerate(evaluate_index_dict.items()):
            x = list(range(len(value)))
            axes[i].plot(x, value, line_styles[i])
            axes[i].set_xlabel("Epoch")
            axes[i].set_title(key)

        # 保存图片
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        plt.savefig(os.path.join(self.model_save_path, "evaluate_index.png"))
        plt.show()
        plt.close()
        print(f"评估指标曲线绘制成功!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--render_mode', type=str, default='human', help='渲染模式')
    parser.add_argument('--model_version', type=str, default='v0', help='模型版本')

    # 不同模型版本
    model_version = {
        "v0": {
            "epochs": 30, "buffer_size": 50, "batch_size": 32,
            "lr": 0.001, "gamma": 0.7, "epsilon": 1,
            "epsilon_decay_step": 30, "epsilon_min": 0.5,
            "target_update_interval": 10, "epoch_print_interval": 10, "seed": 43,
            "model_version": "v0",
        },
        "v1": {
            "epochs": 500, "buffer_size": 50, "batch_size": 32,
            "lr": 0.001, "gamma": 0.7, "epsilon": 1,
            "epsilon_decay_step": 500, "epsilon_min": 0.5,
            "target_update_interval": 10, "epoch_print_interval": 10, "seed": 43,
            "model_version": "v1",
        },
        "v2": {
            "epochs": 500, "buffer_size": 50, "batch_size": 32,
            "lr": 0.001, "gamma": 0.7, "epsilon": 0.9,
            "epsilon_decay_step": 500, "epsilon_min": 0.5,
            "target_update_interval": 10, "epoch_print_interval": 10, "seed": 43,
            "model_version": "v2",
        },
        "v3": {
            "epochs": 50000, "buffer_size": 50, "batch_size": 32,
            "lr": 0.001, "gamma": 0.7, "epsilon": 1,
            "epsilon_decay_step": 50000, "epsilon_min": 0.01,
            "target_update_interval": 200, "epoch_print_interval": 1000, "seed": 43,
            "model_version": "v3",
        },
        "v4": {
            "epochs": 50000, "buffer_size": 100, "batch_size": 32,
            "lr": 0.001, "gamma": 0.7, "epsilon": 1,
            "epsilon_decay_step": 50000, "epsilon_min": 0.5,
            "target_update_interval": 200, "epoch_print_interval": 1000, "seed": 43,
            "model_version": "v4",
        },
    }

    # 训练模型选择
    model_select = parser.parse_args().model_version
    if model_select not in model_version:
        print(f"模型版本选择错误, 请重新选择! 可选版本: {model_version.keys()}")
        exit(0)

    begin_time = time.time()
    # 创建环境
    if parser.parse_args().render_mode == "None":
        from agent_flag_v1 import Environment
    elif parser.parse_args().render_mode == "human":
        from agent_flag_v2 import Environment
    else:
        print(f"渲染模式选择错误, 请重新选择! 可选模式: None, human")
        exit(0)

    env = Environment(n_agents=4,seed=model_version[model_select]["seed"])

    # 运行qmix_Info
    train_begin_time = time.time()
    qmix_algo = QMIX_algo(env=env, epochs=model_version[model_select]["epochs"],
                          buffer_size=model_version[model_select]["buffer_size"],
                          batch_size=model_version[model_select]["batch_size"],
                          lr=model_version[model_select]["lr"],
                          gamma=model_version[model_select]["gamma"],
                          epsilon=model_version[model_select]["epsilon"],
                          epsilon_decay_step=model_version[model_select]["epsilon_decay_step"],
                          epsilon_min=model_version[model_select]["epsilon_min"],
                          target_update_interval=model_version[model_select]["target_update_interval"],
                          model_version=model_version[model_select]["model_version"],
                          epoch_print_interval=model_version[model_select]["epoch_print_interval"],
                          seed=model_version[model_select]["seed"],)
    qmix_algo.run()

    # 运行时间
    end_time = time.time()
    print(f"训练时间:{end_time-train_begin_time:.2f}s")
    print(f"程序运行时间:{time.time()-begin_time:.2f}s")

    # 保存信息
    with open(os.path.join(qmix_algo.model_save_path, "train_info_time.txt"), "w") as f:
        f.write(f"训练时间:{end_time-train_begin_time:.2f}s\n")
        f.write(f"程序运行时间:{time.time()-begin_time:.2f}s")
