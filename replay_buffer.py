import random

import numpy as np
import torch

def batch_observations(observations , device):
    # 输入原始观测；先沿 batch 维拼接，再按 TMRL 约定归一化
    scales = (1000.0 , 10.0 , 10000.0 , 256.0 , 1.0 , 1.0)
    return tuple(torch.as_tensor(np.stack(parts) , dtype = torch.float32 , device = device) / scale for parts , scale in zip(zip(*observations) , scales))

class ReplayBuffer:
    def __init__(self , capacity):
        if capacity < 1:
            raise ValueError("回放池容量必须大于零。")
        self.capacity = capacity
        self.transitions = []
        self.position = 0

    def getlen(self):
        return len(self.transitions)

    def add(self , observation , action , reward , next_observation , terminated , truncated):
        # 每次保存一条原始 transition；复制数组，避免环境复用数组时改写历史
        # 图像在 CPU 上保留 uint8，抽样时才转 float32，减少内存占用
        observation = tuple(np.array(part , dtype = np.uint8 if index == 3 else np.float32 , copy = True) for index , part in enumerate(observation))
        next_observation = tuple(np.array(part , dtype = np.uint8 if index == 3 else np.float32 , copy = True) for index , part in enumerate(next_observation))
        transition = (observation , np.array(action , dtype = np.float32 , copy = True) , float(reward) , next_observation , bool(terminated) , bool(truncated))

        # 容量未满时追加，满了以后循环覆盖最旧的一条
        if len(self.transitions) < self.capacity:
            self.transitions.append(transition)
        else:
            self.transitions[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self , batch_size , device = "cpu"):
        # 均匀、无放回地抽取一批 transition；不足一批时由调用方先继续采集
        samples = random.sample(self.transitions , batch_size)
        observations , actions , rewards , next_observations , terminated , truncated = zip(*samples)

        return (
            batch_observations(observations , device) ,
            torch.as_tensor(np.stack(actions) , dtype = torch.float32 , device = device) ,
            torch.tensor(rewards , dtype = torch.float32 , device = device) ,
            batch_observations(next_observations , device) ,
            torch.tensor(terminated , dtype = torch.bool , device = device) ,
            torch.tensor(truncated , dtype = torch.bool , device = device)
        )
