import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import torch

from game import GameCollector
from model import ActorNet
from replay_buffer import ReplayBuffer


# 测试参数：一回合最多约 60 秒，CPU 足够运行这个小网络
project_root = Path(__file__).resolve().parent
weights_path = project_root / "weights/pretrained/actor_images.pt"
episode_steps = 1200
startup_delay = 5
device = "cpu"
seed = 0


# 加载自己的 Actor，只做推理
torch.set_num_threads(2)
torch.manual_seed(seed)
actor = ActorNet().to(device)
actor.load_state_dict(torch.load(weights_path , map_location = device , weights_only = True) , strict = True)
actor.eval()

# 复用游戏交互接口，回合结束后保留游戏画面
buffer = ReplayBuffer(episode_steps)
collector = GameCollector(episode_steps , device , seed , startup_delay , wait_on_done = False)
run_dir = project_root / "results/test" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
run_dir.mkdir(parents = True , exist_ok = False)
summary = {"weights": str(weights_path) , "device": device , "deterministic": True , "status": "running"}
started = time.perf_counter()
print("测试权重:" , weights_path , flush = True)
print("结果目录:" , run_dir , flush = True)

with warnings.catch_warnings(record = True) as warning_records:
    warnings.simplefilter("always")
    try:
        # test=True 使用 tanh(mean)，跳过随机动作预热
        summary.update(collector.collect_episode(actor , buffer , warmup_steps = 0 , test = True))
        summary["status"] = "completed"
    except KeyboardInterrupt:
        summary["status"] = "stopped"
        print("测试已停止。" , flush = True)
    except Exception as error:
        summary["status"] = "failed"
        summary["error"] = repr(error)
        raise
    finally:
        # 先释放手柄，再保存动作、速度和奖励；不写训练检查点
        collector.close()
        summary["elapsed_seconds"] = time.perf_counter() - started
        summary["steps"] = buffer.getlen()
        summary["reward"] = sum(sample[2] for sample in buffer.transitions)
        summary["max_speed"] = max((float(sample[3][0][0]) for sample in buffer.transitions) , default = 0.0)
        summary["time_step_timeouts"] = sum("Time-step timed out" in str(item.message) for item in warning_records)
        trace = [{"step": index , "action": action.tolist() , "reward": reward , "speed": float(next_observation[0][0]) , "terminated": terminated , "truncated": truncated}
                 for index , (observation , action , reward , next_observation , terminated , truncated) in enumerate(buffer.transitions)]
        (run_dir / "test.json").write_text(json.dumps({"summary": summary , "steps": trace} , ensure_ascii = False , indent = 2 , allow_nan = False) , encoding = "utf-8")
        print(json.dumps(summary , ensure_ascii = False) , flush = True)
