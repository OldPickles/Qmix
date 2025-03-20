this is first version for Qmix
add function:
	1. 变更环境为：agent和shovels之间协作的环境。
	    - 铲子和智能体随机运动，不考虑个体级别的行为协同，仅考虑通过q_mix_value最小化获取的协作约束。
	    - 铲子与智能体使用同一个智能体网络产生动作。
	2. 变更 delay_buffer 使其经验条可以存储env中的所有智能体的经验。
	    self.n_workers = env.n_agents -> self.n_workers = env.n_agents * env.n_shovels
enlightenment：
1. 以整体的reward作为考察每个worker的行为的准则，会消耗掉每个worker单独的探索所做出的贡献。
    - 考虑每个worker分配一个训练网络，然后以固定的更新频率将当前效率最好的训练网络复制到其他网络中。
2. 考虑worker的网络训练包含两个阶段：第一阶段仅仅训练worker的行为是贪婪的。第二阶段再通过mix_net训练出智能体之间协作的约束。
