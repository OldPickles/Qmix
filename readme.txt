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
3. 行为的值选择不再是[0,1,2,3,4] 来代表上下左右以及静止。而是考虑直接给出接下来所处位置的索引。


1. onehot编码
2. 行为选择， 加一个不动。
3. 规模。
4. 硬策略，过滤行为，
5. 规模缩小。
6. 奖励类别变小。


questions:
1. 昨日未完：没有将wall的相关变量全部改变为obstacles ok
2. 修改环境为服务器可运行模式。
3. 修改环境为gym环境。

