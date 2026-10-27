import json
import random
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from game import GameCollector
from model import ActorNet , CriticNet
from replay_buffer import ReplayBuffer
from training_io import load_checkpoint , save_checkpoint

# 训练参数
num_episodes = 100
episode_steps = 200
buffer_capacity = 10000
batch_size = 32
warmup_steps = 256
updates_per_episode = 100
seed = 0

# SAC 参数，保持原 loss.py 中的设置
gamma = 0.995
alpha = 0.01

# 运行参数
startup_delay = 5
resume_path = None # 自己保存的 checkpoint.pt 路径；恢复时重新填充回放池
project_root = Path(__file__).resolve().parent
device = "cuda" if torch.cuda.is_available() else "cpu"

assert 1 <= batch_size <= warmup_steps <= buffer_capacity
assert num_episodes > 0 and episode_steps > 0 and updates_per_episode > 0
random.seed(seed)
torch.manual_seed(seed)
torch.set_num_threads(2)

# 显式创建 Actor、两个独立 Critic 和对应的目标网络
config_path = project_root / "configs" / "tmrl.json"
config = json.loads(config_path.read_text(encoding = "utf-8-sig"))["ALG"]
polyak = config["POLYAK"]
actor = ActorNet().to(device)
critic1 = CriticNet().to(device)
critic2 = CriticNet().to(device)
target_critic1 = deepcopy(critic1).requires_grad_(False)
target_critic2 = deepcopy(critic2).requires_grad_(False)

# Actor 和 Critic 分别使用自己的优化器
critic_parameters = list(critic1.parameters()) + list(critic2.parameters())
actor_optimizer = torch.optim.Adam(actor.parameters() , lr = config["LR_ACTOR"] , betas = tuple(config["BETAS_ACTOR"]))
critic_optimizer = torch.optim.Adam(critic_parameters , lr = config["LR_CRITIC"] , betas = tuple(config["BETAS_CRITIC"]))

# 这些字典仅用于保存和恢复，训练中仍直接使用各网络和优化器
models = {"actor": actor , "critic1": critic1 , "critic2": critic2 , "target_critic1": target_critic1 , "target_critic2": target_critic2}
optimizers = {"actor_optimizer": actor_optimizer , "critic_optimizer": critic_optimizer}
buffer = ReplayBuffer(buffer_capacity)
initial_steps = 0
total_updates = 0
episodes_completed = 0

if resume_path is not None:
    initial_steps , total_updates , episodes_completed = load_checkpoint(resume_path , models , optimizers , device)
    print("恢复检查点:" , resume_path , "回放池将重新采集。")

run_dir = project_root / "results" / "training" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
run_dir.mkdir(parents = True , exist_ok = False)
checkpoint_path = run_dir / "checkpoint.pt"
collector = GameCollector(episode_steps , device , seed , startup_delay , initial_steps)
print("current device:" , device)
print("保存目录:" , run_dir)

try:
    for episode in range(episodes_completed , episodes_completed + num_episodes):
        # 获取一回合新数据；返回时已结束采集，可以进行训练
        episode_data = collector.collect_episode(actor , buffer , warmup_steps)
        episodes_completed = episode + 1
        episode_updates = 0
        total_critic_loss = 0.0
        total_actor_loss = 0.0

        # 与监督学习一样，反复取 batch、计算 loss、反向传播并更新参数
        actor.train()
        if buffer.getlen() >= warmup_steps:
            for update in range(updates_per_episode):
                collector.check_stop()
                observation , action , reward , next_observation , terminated , truncated = buffer.sample(batch_size , device)
                actor_optimizer.zero_grad(set_to_none = True)
                critic_optimizer.zero_grad(set_to_none = True)

                # 计算 Critic 的目标；截断仍保留后续价值，真正终止才屏蔽
                with torch.no_grad():
                    next_action , next_log_prob = actor.sample(*next_observation)
                    next_q1 = target_critic1(*next_observation , next_action)
                    next_q2 = target_critic2(*next_observation , next_action)
                    next_q = torch.minimum(next_q1 , next_q2)
                    target = reward + gamma * (1 - terminated.float()) * (next_q - alpha * next_log_prob)

                # 用回放中实际记录的动作训练两个 Critic
                q1 = critic1(*observation , action)
                q2 = critic2(*observation , action)
                loss_critic = F.mse_loss(q1 , target) + F.mse_loss(q2 , target)
                loss_critic.backward()
                critic_optimizer.step()

                # 冻结 Critic 参数，但保留 Q 对 Actor 生成动作的梯度
                critic_optimizer.zero_grad(set_to_none = True)
                for parameter in critic_parameters:
                    parameter.requires_grad_(False)
                try:
                    policy_action , log_prob = actor.sample(*observation)
                    q1 = critic1(*observation , policy_action)
                    q2 = critic2(*observation , policy_action)
                    q = torch.minimum(q1 , q2)
                    loss_actor = torch.mean(alpha * log_prob - q)
                    loss_actor.backward()
                    actor_optimizer.step()
                finally:
                    for parameter in critic_parameters:
                        parameter.requires_grad_(True)

                # 目标网络按旧值 0.995、新值 0.005 的比例缓慢更新
                with torch.no_grad():
                    for critic , target_critic in ((critic1 , target_critic1) , (critic2 , target_critic2)):
                        for parameter , target_parameter in zip(critic.parameters() , target_critic.parameters()):
                            target_parameter.mul_(polyak).add_(parameter , alpha = 1 - polyak)

                total_critic_loss += loss_critic.item()
                total_actor_loss += loss_actor.item()
                episode_updates += 1
                total_updates += 1

        metrics = {
            "episode": episodes_completed , **episode_data ,
            "buffer_size": buffer.getlen() , "total_steps": collector.total_steps , "total_updates": total_updates ,
            "updates": episode_updates ,
            "critic_loss": total_critic_loss / episode_updates if episode_updates else None ,
            "actor_loss": total_actor_loss / episode_updates if episode_updates else None
        }
        with (run_dir / "history.jsonl").open("a" , encoding = "utf-8") as file:
            file.write(json.dumps(metrics , ensure_ascii = False , allow_nan = False) + "\n")
        print("episode:" , episodes_completed , "reward:" , episode_data["reward"] , "buffer:" , buffer.getlen() , "updates:" , episode_updates , "loss:" , metrics["critic_loss"] , metrics["actor_loss"] , flush = True)
        save_checkpoint(checkpoint_path , models , optimizers , collector.total_steps , total_updates , episodes_completed)

except KeyboardInterrupt:
    print("停止训练，正在保存模型。")
finally:
    try:
        collector.close()
    finally:
        save_checkpoint(checkpoint_path , models , optimizers , collector.total_steps , total_updates , episodes_completed)
        print("model saved:" , checkpoint_path)
