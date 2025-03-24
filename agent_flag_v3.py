import itertools
import random

import numpy as np
import tkinter as tk
import time

import torch
from PIL import ImageTk, Image


class Environment(tk.Tk):
    def __init__(self, render_mode="None", n_agents=5, agent_vision_length=5, padding=5, width=5, height=5, seed=43):
        """
        初始化环境
        :param render_mode:
            None: 不渲染
            human: 人类可视化
        :param n_agents:
        :param agent_vision_length:
        :param padding:
        :param width:
        :param height:
        :param seed:
        """
        super(Environment, self).__init__()

        self.render_mode = render_mode
        self.seed = seed
        self.walls_position = []
        self.canvas = None
        self.pixels = 20
        self.width = width  # 可活动的20x20的网格
        self.height = height
        self.padding = padding
        self.WIDTH = self.width + self.padding * 2
        self.HEIGHT = self.height + self.padding * 2

        self.canvas = tk.Canvas(self, bg='white',
                                height=self.HEIGHT * self.pixels,
                                width=self.WIDTH * self.pixels)
        self.canvas.pack()
        self.mode_update()

        self.n_agents = n_agents
        self.n_flag = int(self.n_agents * 4)
        self.n_shovels = n_agents
        self.n_obstacle = n_agents * 10

        self.n_objects = 5 # roads, agents，shovels，flag，obstacle
        self.n_workers = self.n_agents + self.n_shovels

        self.avail_actions_shape = (5,)
        self.avail_actions_dim = 5
        self.action_shape = (1,)
        self.action_dim = 1
        self.action_values = [0, 1, 2, 3, 4]
        self.action_value_info = {
            "no_op": 0,
            "up": 1,
            "down": 2,
            "left": 3,
            "right": 4,
        }

        self.agent_vision_length = min(agent_vision_length, self.padding)
        self.observation_shape = (self.n_objects, 1 + self.agent_vision_length * 2, 1 + self.agent_vision_length * 2)
        self.observation_dim = self.observation_shape[0] * self.observation_shape[1] * self.observation_shape[2]
        self.observation_values = [0, 1]

        self.state_shape = (self.n_objects, self.WIDTH, self.HEIGHT)
        self.state_dim = self.state_shape[0] * self.state_shape[1] * self.state_shape[2]
        self.state_values = [0, 1]
        self.state_value_index = {
            "obstacle": 0,
            "agent": 1,
            "shovel": 2,
            "flag": 3,
        }
        self.state_value_info = {
            "road": 1,
            "obstacle": 1,
            "agent": 1,
            "shovel": 1,
            "flag": 1,
        }

        # 走到相应位置获得得奖励
        self.reward_info = {
            "reach_flag": 1000,
            "reach_obstacle": -100,
            "reach_road": -1,
        }
        self.reward_shape = (1,)
        self.reward_dim = 1
        self.reward_values = self.reward_info.values()

        self.done_shape = (1,)
        self.done_dim = 1
        self.done_values = [1, 0]    # 0: 未结束， 1: 结束

        self.position_shape = (2,)
        WIDTH_set = set(list(range(self.WIDTH)))
        HEIGHT_LIST = list(range(self.HEIGHT))
        WIDTH_set.update(HEIGHT_LIST)
        self.position_values = list(WIDTH_set)

        # 环境空间占用情况，初始化为全部都是空地
        self.space_occupy = np.full((self.n_objects, self.WIDTH, self.HEIGHT),
                                    self.state_value_info['road'], dtype=int)
                            # 共有5个矩阵，对应位置放在self.state_value_info中

        # 记录可用空地
        self.avail_positions = list(itertools.product(range(self.padding, self.padding + self.width),
                                                      range(self.padding, self.padding + self.height)))
        self.avail_positions_original = self.avail_positions.copy()


        # 记录元素对象
        self.flags = []
        self.agents = []
        self.obstacles = []
        self.shovels = []
        self.walls = []

        # 记录元素的位置
        self.flag_positions = []
        self.agent_positions = []
        self.obstacle_positions = []
        self.shovel_positions = []
        self.walls_positions = []

        self.tk_photo_objects = []  # 显式存储tk_photo图像文件

        # 将边界self.padding设置为障碍物
        wall_file_path = "./images/tree.png"
        self.wall_object = ImageTk.PhotoImage(Image.open(wall_file_path).resize((self.pixels, self.pixels)))
        self.set_boundary()

        # 初始化元素
        img_flag_path = 'images/flag.png'
        img_agent_path = 'images/agent.png'
        img_obstacle_path = 'images/tree.png'
        img_shovel_path = 'images/shovel.png'
        self.flag_object = ImageTk.PhotoImage(Image.open(img_flag_path).resize((self.pixels, self.pixels)))
        self.agent_object = ImageTk.PhotoImage(Image.open(img_agent_path).resize((self.pixels, self.pixels)))
        self.obstacle_object = ImageTk.PhotoImage(Image.open(img_obstacle_path).resize((self.pixels, self.pixels)))
        self.shovel_object = ImageTk.PhotoImage(Image.open(img_shovel_path).resize((self.pixels, self.pixels)))
        self.build_environment()

        # 记录最初的占用记录
        self.space_occupy_original = self.space_occupy.copy()

        self.mode_update()

    def build_environment(self):
        # 建立网格
        for column in range(0, self.WIDTH * self.pixels, self.pixels):
            x0, y0, x1, y1 = column, 0, column, self.HEIGHT * self.pixels
            self.canvas.create_line(x0, y0, x1, y1, fill='grey')

        for row in range(0, self.HEIGHT * self.pixels, self.pixels):
            x0, y0, x1, y1 = 0, row, self.WIDTH * self.pixels, row
            self.canvas.create_line(x0, y0, x1, y1, fill='grey')

        # 添加元素
        self.init_element()

        self.canvas.pack()

    def set_boundary(self):
        """
        将边界 padding=2 设置为障碍物
        :return:
        """
        # 填充墙壁
        # 上边界
        self.walls_position += list(itertools.product(range(self.padding), range(self.WIDTH)))
        # 下边界
        self.walls_position += list(itertools.product(range(self.HEIGHT-self.padding, self.HEIGHT), range(self.WIDTH)))
        # 左边界
        self.walls_position += list(itertools.product(range(self.padding, self.HEIGHT - self.padding),
                                                      range(self.padding)))
        # 右边界
        self.walls_position += list(itertools.product(range(self.padding, self.HEIGHT - self.padding),
                                                      range(self.WIDTH - self.padding, self.WIDTH)))

        # 绘制墙壁
        for _ in self.walls_position:
            tk_photo = self.canvas.create_image(self.pixels * _[0], self.pixels * _[1], anchor='nw',
                                                image=self.wall_object)
            self.tk_photo_objects.append(tk_photo)
            self.walls.append([tk_photo, _])
            self.walls_positions.append(_)
            self.space_occupy[self.state_value_index['obstacle'], _[0], _[1]] = self.state_value_info['wall']
        self.mode_update()

    def generate_random_position(self):
        """
        随机生成一个已占用之外的位置
        :return: list [x, y]
        """
        random.seed(self.seed)
        if len(self.avail_positions) > 0:
            position = random.sample(self.avail_positions, 1)[0]
            return position
        else:
            return None

    def init_element(self):
        """
        添加元素
        :return:
        """
        # 添加障碍物
        for _ in range(self.n_obstacle):
            obstacle_position = self.generate_random_position()
            self.space_occupy[obstacle_position[0], obstacle_position[1]] = self.state_value_info['obstacle']
            tk_photo = self.canvas.create_image(self.pixels * obstacle_position[0],
                                                self.pixels * obstacle_position[1],
                                                anchor='nw', image=self.obstacle_object)
            self.tk_photo_objects.append(tk_photo)
            self.obstacles.append([tk_photo, obstacle_position])
            self.obstacle_positions.append(obstacle_position)
            self.space_occupy[self.state_value_index['obstacle'], obstacle_position[0], obstacle_position[1]] = \
            self.state_value_info['obstacle']

        # 添加旗子
        for _ in range(self.n_flag):
            flag_position = self.generate_random_position()
            self.space_occupy[flag_position[0], flag_position[1]] = self.state_value_info['flag']
            tk_photo = self.canvas.create_image(self.pixels * flag_position[0],
                                                self.pixels * flag_position[1],
                                                anchor='nw', image=self.flag_object)
            self.tk_photo_objects.append(tk_photo)
            self.flags.append([tk_photo, flag_position])
            self.flag_positions.append(flag_position)
            self.space_occupy[self.state_value_index['flag'], flag_position[0], flag_position[1]] = self.state_value_info['flag']

        # 添加智能体
        for _ in range(self.n_agents):
            agent_position = self.generate_random_position()
            self.space_occupy[agent_position[0], agent_position[1]] = self.state_value_info['agent']
            tk_photo = self.canvas.create_image(self.pixels * agent_position[0],
                                                self.pixels * agent_position[1],
                                                anchor='nw', image=self.agent_object)
            self.tk_photo_objects.append(tk_photo)
            self.agents.append([tk_photo, agent_position])
            self.agent_positions.append(agent_position)
            self.space_occupy[self.state_value_index['agent'], agent_position[0], agent_position[1]] = self.state_value_info['agent']

        # 添加铲子
        for _ in range(self.n_shovels):
            shovel_position = self.generate_random_position()
            self.space_occupy[shovel_position[0], shovel_position[1]] = self.state_value_info['shovel']
            tk_photo = self.canvas.create_image(self.pixels * shovel_position[0],
                                                self.pixels * shovel_position[1],
                                                anchor='nw', image=self.shovel_object)
            self.tk_photo_objects.append(tk_photo)
            self.shovels.append([tk_photo, shovel_position])
            self.shovel_positions.append(shovel_position)
            self.space_occupy[self.state_value_index['shovel'], shovel_position[0], shovel_position[1]] = self.state_value_info['shovel']

    def render(self):
        """
        刷新环境
        :return:
        """
        self.update()

    def reset(self):
        """
        重置环境
        :return:
        """
        # 删除所有智能体图像，旗子图像和铲子图像
        for agent in self.agents:
            self.canvas.delete(agent[0])
        for flag in self.flags:
            self.canvas.delete(flag[0])
        for shovel in self.shovels:
            self.canvas.delete(shovel[0])

        self.agents.clear()
        self.flags.clear()
        self.shovels.clear()

        self.agent_positions.clear()
        self.shovel_positions.clear()
        self.flag_positions.clear()

        self.tk_photo_objects = self.tk_photo_objects[: (len(self.walls) + len(self.obstacles))]

        # 按照self.space_occupy_original重新添加智能体、旗子和铲子
        self.space_occupy = self.space_occupy_original.copy()
        for (key, value) in self.state_value_index.items():
            if key == 'agent' or key == 'flag' or key =='shovel':
                for (i, j) in itertools.product(range(self.WIDTH), range(self.HEIGHT)):
                    if self.space_occupy[value, i, j] == self.state_value_info[key]:
                        tk_photo = self.canvas.create_image(self.pixels * i, self.pixels * j, anchor='nw',
                                                            image=eval(f'self.{key}_object'))
                        self.tk_photo_objects.append(tk_photo)
                        eval(f'self.{key}s.append([tk_photo, [i, j]])')
                        eval(f'self.{key}_positions.append([i, j])')
                        self.space_occupy[value, i, j] = self.state_value_info[key]
                    else:
                        pass
            else:
                pass
        self.mode_update()

    def step(self, actions):
        """
        执行动作
        :param actions: shape = (n_workers, 1)
            前半部分时智能体的动作，后半部分是铲子的动作
        :return:
            dones: bool
            rewards: np.array
                    元素类型：int
                    shape：(n_agents, 1)
                    元素含义：每一个智能体获得的奖励
        """
        # 此2个列表作为step返回的值，长度均等于n_agents + n_shovels
        dones = []
        rewards = []

        # 智能体的行为
        for agent_index in range(len(self.agents)):
            agent_position = self.agents[agent_index][1].copy()
            action = actions[agent_index]

            # 执行动作
            if action == self.action_value_info['up']:
                self.space_occupy[self.state_value_index['agent'],agent_position[0], agent_position[1]] = self.state_value_info['road']
                agent_position[1] -= 1
            if action == self.action_value_info['down']:
                self.space_occupy[self.state_value_index['agent'],agent_position[0], agent_position[1]] = self.state_value_info['road']
                agent_position[1] += 1
            if action == self.action_value_info['left']:
                self.space_occupy[self.state_value_index['agent'],agent_position[0], agent_position[1]] = self.state_value_info['road']
                agent_position[0] -= 1
            if action == self.action_value_info['right']:
                self.space_occupy[self.state_value_index['agent'],agent_position[0], agent_position[1]] = self.state_value_info['road']
                agent_position[0] += 1

            # 更新智能体位置
            self.canvas.delete(self.agents[agent_index][0])
            self.agents[agent_index][1] = agent_position.copy()  # 这里没有赋值，假定获得agent_position改变那么，在self.agents
            # 如果碰到障碍物，则该旗子的回合结束
            if self.space_occupy[self.state_value_index['obstacle'], agent_position[0], agent_position[1]] == self.state_value_info['obstacle']:
                dones.append(True)
                rewards.append(self.reward_info['reach_obstacle'])
            # 如果碰到旗子，则获得奖励
            elif self.space_occupy[self.state_value_index['flag'], agent_position[0], agent_position[1]] == self.state_value_info['flag']:
                dones.append(False)
                rewards.append(self.reward_info['reach_flag'])
            # 如果碰到墙，则该回合结束
            elif self.space_occupy[self.state_value_index['obstacle'], agent_position[0], agent_position[1]] == self.state_value_info['obstacle']:
                dones.append(True)
                rewards.append(self.reward_info['reach_obstacle'])
            # 到达空地
            else:
                dones.append(False)
                rewards.append(self.reward_info['reach_road'])
            # 存储当前位置，添加图像
            tk_photo = self.canvas.create_image(self.pixels * agent_position[0], self.pixels * agent_position[1],
                                                anchor='nw', image=self.agent_object)
            self.tk_photo_objects.append(tk_photo)
            self.agents[agent_index] = [tk_photo, agent_position]
            self.mode_update()

        # 铲子的行为
        for shovel_index in range(len(self.shovels)):
            shovel_position = self.shovels[shovel_index][1].copy()
            action = actions[len(self.agents) + shovel_index]
            # 执行动作
            if action == self.action_value_info['up']:
                self.space_occupy[self.state_value_index['shovel'], shovel_position[0], shovel_position[1]] = self.state_value_info['road']
                shovel_position[1] -= 1
            if action == self.action_value_info['down']:
                self.space_occupy[self.state_value_index['shovel'], shovel_position[0], shovel_position[1]] = self.state_value_info['road']
                shovel_position[1] += 1
            if action == self.action_value_info['left']:
                self.space_occupy[self.state_value_index['shovel'], shovel_position[0], shovel_position[1]] = self.state_value_info['road']
                shovel_position[0] -= 1
            if action == self.action_value_info['right']:
                self.space_occupy[self.state_value_index['shovel'], shovel_position[0], shovel_position[1]] = self.state_value_info['road']
                shovel_position[0] += 1

            # 更新铲子位置
            self.canvas.delete(self.shovels[shovel_index][0])
            # 如果铲子移动到障碍物上，则该回合结束
            if self.space_occupy[self.state_value_index['obstacle'], shovel_position[0], shovel_position[1]] == self.state_value_info['obstacle']:
                dones.append(True)
                rewards.append(self.reward_info['reach_obstacle'])
            # 如果铲子移动到旗子上，则获得奖励
            elif self.space_occupy[self.state_value_index['flag'], shovel_position[0], shovel_position[1]] == self.state_value_info['flag']:
                dones.append(False)
                rewards.append(self.reward_info['reach_flag'])
                # 如果这里也有智能体，则奖励加倍，且删除该位置的旗子
                for agent in self.agents:
                    if agent[1] == shovel_position:
                        rewards[-1] *= 2
                        self.canvas.delete(self.flags[self.flag_positions.index(shovel_position)][0])
                        self.tk_photo_objects.remove(self.flags[self.flag_positions.index(shovel_position)][0])
                        self.flags.remove(self.flags[self.flag_positions.index(shovel_position)])
                        self.flag_positions.remove(shovel_position)
                        self.space_occupy[self.state_value_index['shovel'], shovel_position[0], shovel_position[1]] = self.state_value_info['road']
                        self.space_occupy[self.state_value_index['flag'], shovel_position[0], shovel_position[1]] = self.state_value_info['road']
            # 如果铲子移动到墙上，则该回合结束
            elif self.space_occupy[self.state_value_index['wall'], shovel_position[0], shovel_position[1]] == self.state_value_info['wall']:
                dones.append(True)
                rewards.append(self.reward_info['reach_wall'])
            # 到达空地
            else:
                dones.append(False)
                rewards.append(self.reward_info['reach_road'])
            # 添加图像
            tk_photo = self.canvas.create_image(self.pixels * shovel_position[0], self.pixels * shovel_position[1],
                                                anchor='nw', image=self.shovel_object)
            self.tk_photo_objects.append(tk_photo)
            self.shovels[shovel_index] = [tk_photo, shovel_position]
            self.mode_update()

        # 检查是否找到所有旗子，如果全都找到，则所有智能体的回合结束
        if len(self.flags) == 0:
            dones = [True for _ in range(len(self.agents)+len(self.shovels))]
        return (torch.tensor(np.array(dones).reshape((self.n_agents+self.n_shovels,) + self.done_shape), dtype=torch.float32),
                torch.tensor(np.array(rewards).reshape((self.n_agents+self.n_shovels,) + self.reward_shape), dtype=torch.float32),
                )

    def actions_sample(self, avail_actions=None):
        """
        随机采样动作
        :param :
        avail_actions: 类型为np.array, shape=(self.n_workers, 4)
            代表智能体上下左右四个方向是否可行，若若有road则为1，否则为0
            注意： 此处的n_workers必须是初始的n_agents+n_shovels
        :return:
        """
        actions = []
        if avail_actions is None:
            for _ in range(self.n_workers):
                actions.append(random.randint(0, self.avail_actions_dim-1))
            return np.array(actions).reshape(self.n_workers, 1)
        else:
            for _ in avail_actions:
                # 如果有路则随机选择一个可行的方向
                if sum(_) > 0:
                    actions.append(random.choice(np.where(_ == 1)[0]))
                else:
                    actions.append(random.randint(0, self.avail_actions_dim-1))
            return np.array(actions).reshape(self.n_workers, 1)

    def get_observations(self, ):
        """
        获取所有智能体的观测值
        :return:
        observations, 类型为np.array, shape=(self.n_agents, (self.agent_vision_length*2+1)**2)
            代表着以智能体为中心环视一周，长度为self.agent_vision_length
        """
        observations = []
        workers = self.agents + self.shovels

        # 工人的观测值
        for worker in workers:
            worker_position = worker[1]
            up_line = worker_position[1] - self.agent_vision_length
            down_line = worker_position[1] + self.agent_vision_length + 1
            left_line = worker_position[0] - self.agent_vision_length
            right_line = worker_position[0] + self.agent_vision_length + 1
            observation = self.space_occupy[:, up_line:down_line, left_line:right_line]
            observation = observation.tolist()
            observations.append(observation)
        return torch.tensor(np.array(observations), dtype=torch.float32)

    def get_state(self, ):
        """
        获取全局状态
        :return:
        state, 类型为np.array, shape=(self.WIDTH, self.HEIGHT)
        """
        return torch.tensor(self.space_occupy.copy(), dtype=torch.float32)

    def get_avail_actions(self, ):
        """
        获取所有智能体和铲子的可用动作
        :return:
        actions, 类型为np.array, shape=(self.n_agents, 4)
            代表智能体上下左右四个方向是否可行，若若有road则为1，否则为0
        """
        actions = []
        workers = self.agents + self.shovels

        for worker in workers:
            position = worker[1]
            action = []
            for move_x, move_y in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                if self.space_occupy[self.state_value_index['road'], position[0] + move_x, position[1] + move_y] == 0:
                    action.append(1)
                else:
                    action.append(0)
            actions.append(action)
        return torch.tensor(np.array(actions), dtype=torch.float32)

    def is_done(self, ):
        """
        判断是否所有旗子都找到了
        :return:
            done, 类型为bool, 若所有旗子都找到了则为True，否则为False
        """
        if len(self.flags) == 0:
            return True
        else:
            return False

    def mode_update(self, ):
        """
        更新模式
        :return:
        """
        if self.render_mode == 'human':
            self.update()
        elif self.render_mode == "None":
            pass
        else:
            raise ValueError("render_mode must be 'human' or 'None'")



if __name__ == '__main__':
    env = Environment(n_agents=4, render_mode="human")
    epoch = 0
    while True:
        epoch += 1
        print(f'epoch: {epoch}', end='\t')
        iteration = 0
        env.reset()
        while True:
            actions = env.actions_sample()
            dones, rewards, next_positions = env.step(actions)
            iteration += 1
            time.sleep(0.1)
            if any(dones):
                print(f'iteration: {iteration}')
                break

