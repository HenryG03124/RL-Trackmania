import argparse
import json
import random
import sys
import time
from pathlib import Path
from copy import deepcopy

import torch

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0 , str(project_root))

from replay_buffer import ReplayBuffer
from model import ActorNet , CriticNet
from loss import critic_loss , actor_loss


def main():
    parser = argparse.ArgumentParser(description = "用已有记录离线检查网络、回放与 loss.py 对照实现，不操作游戏、不保存模型。")
    parser.add_argument("--recording" , type = Path , required = True)
    parser.add_argument("--batch-size" , type = int , default = 32)
    parser.add_argument("--steps" , type = int , default = 2)
    parser.add_argument("--device" , choices = ("cpu" , "cuda") , default = "cpu")
    args = parser.parse_args()
    if not 1 <= args.steps <= 20 or not 1 <= args.batch_size <= 256:
        parser.error("该短检查要求 steps 在 1 到 20 之间，batch-size 在 1 到 256 之间。")

    random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(2)
    buffer = ReplayBuffer(capacity = 10000)
    summary = None
    loaded = 0

    # 复用阶段 0 的原始 JSONL，末尾的 summary 不是 transition
    with args.recording.open(encoding = "utf-8-sig") as file:
        for line in file:
            record = json.loads(line)
            if record["type"] == "transition":
                buffer.add(
                    record["obs"] , record["action"] , record["reward"] ,
                    record["next_obs"] , record["terminated"] , record["truncated"]
                )
                loaded += 1
            elif record["type"] == "summary":
                summary = record

    if summary is None or summary["interface"] != "TM20IMAGES" or not summary["completed"] or not summary["all_finite"]:
        raise ValueError("需要一份正常结束且数值有效的 TM20IMAGES 采集记录；这不要求赛车完赛。")
    if buffer.getlen() < args.batch_size:
        raise ValueError(f"只有 {buffer.getlen()} 条数据，不足以抽取 {args.batch_size} 条。")

    # 单独检查保留的 loss.py 对照实现；实际训练算法已直接展开在 train.py
    config = json.loads((project_root / "configs" / "tmrl.json").read_text(encoding = "utf-8-sig"))["ALG"]
    device = torch.device(args.device)
    actor = ActorNet().to(device)
    critic1 = CriticNet().to(device)
    critic2 = CriticNet().to(device)
    target_critic1 = deepcopy(critic1).requires_grad_(False)
    target_critic2 = deepcopy(critic2).requires_grad_(False)
    critic_parameters = list(critic1.parameters()) + list(critic2.parameters())
    actor_optimizer = torch.optim.Adam(actor.parameters() , lr = config["LR_ACTOR"] , betas = tuple(config["BETAS_ACTOR"]))
    critic_optimizer = torch.optim.Adam(critic_parameters , lr = config["LR_CRITIC"] , betas = tuple(config["BETAS_CRITIC"]))
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # 只执行少量更新，检查真实记录能否通过用户自己的网络和 loss
    for step in range(args.steps):
        observation , action , reward , next_observation , terminated , truncated = buffer.sample(
            args.batch_size , device = device
        )
        assert action.shape == (args.batch_size , 3)
        assert reward.shape == terminated.shape == truncated.shape == (args.batch_size ,)
        assert observation[3].shape == next_observation[3].shape == (args.batch_size , 4 , 64 , 64)

        if args.device == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        # 截断标记保留用于回合管理；训练目标只由真正终止标记屏蔽
        actor_optimizer.zero_grad(set_to_none = True)
        critic_optimizer.zero_grad(set_to_none = True)
        loss_critic = critic_loss(actor , critic1 , critic2 , target_critic1 , target_critic2 , observation , action , reward , next_observation , terminated)
        loss_critic.backward()
        critic_optimizer.step()

        critic_optimizer.zero_grad(set_to_none = True)
        for parameter in critic_parameters:
            parameter.requires_grad_(False)
        try:
            loss_actor = actor_loss(actor , critic1 , critic2 , observation)
            loss_actor.backward()
            actor_optimizer.step()
        finally:
            for parameter in critic_parameters:
                parameter.requires_grad_(True)

        with torch.no_grad():
            for critic , target_critic in ((critic1 , target_critic1) , (critic2 , target_critic2)):
                for parameter , target_parameter in zip(critic.parameters() , target_critic.parameters()):
                    target_parameter.mul_(config["POLYAK"]).add_(parameter , alpha = 1 - config["POLYAK"])
        logs = {"critic_loss": loss_critic.item() , "actor_loss": loss_actor.item()}
        if args.device == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        print(json.dumps({"step": step + 1 , **logs , "update_seconds": elapsed} , allow_nan = False))

    print(json.dumps({
        "recording": str(args.recording.resolve()) , "loaded_transitions": loaded ,
        "buffer_size": buffer.getlen() , "batch_size": args.batch_size , "updates": args.steps ,
        "device": str(device) ,
        "peak_cuda_memory_mib": torch.cuda.max_memory_allocated(device) / 1024 ** 2 if args.device == "cuda" else None
    } , allow_nan = False))


if __name__ == "__main__":
    main()

