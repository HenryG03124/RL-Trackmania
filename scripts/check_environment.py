import ctypes
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

# 在导入 TMRL 前统一为物理像素，避免 Windows 缩放导致截图出现黑边。
if sys.platform == "win32":
    dpi_enabled = ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    assert dpi_enabled , "Could not enable per-monitor DPI awareness."

import numpy as np
import tmrl
import tmrl.config.config_constants as cfg
from tmrl.config.config_objects import CONFIG_DICT , POLICY , OBS_PREPROCESSOR

project_root = Path(__file__).resolve().parents[1]
with (project_root / "configs/stage0.json").open(encoding = "utf-8") as file:
    config = json.load(file)
run_game = config["run_game"]
max_steps = config["max_steps"]
device = config["device"]
assert isinstance(run_game , bool) and 1 <= max_steps <= 200

# 确认当前接口与官方预训练权重配套。
pretrained_runs = {"TM20LIDAR": "SAC_4_LIDAR_pretrained" , "TM20IMAGES": "SAC_4_imgs_pretrained"}
assert cfg.RTGYM_INTERFACE in pretrained_runs , "Select a supported pretrained interface."
assert cfg.RUN_NAME == pretrained_runs[cfg.RTGYM_INTERFACE] , "Select the matching pretrained checkpoint."
if cfg.RTGYM_INTERFACE == "TM20IMAGES":
    assert cfg.GRAYSCALE and cfg.IMG_WIDTH == 64 and cfg.IMG_HEIGHT == 64
assert cfg.IMG_HIST_LEN == 4 and cfg.ACT_BUF_LEN == 2 and cfg.PRAGMA_GAMEPAD
assert Path(cfg.MODEL_PATH_WORKER).is_file() , cfg.MODEL_PATH_WORKER

if run_game:
    CONFIG_DICT["benchmark"] = True
    CONFIG_DICT["benchmark_type"] = "polyak"
    CONFIG_DICT["ep_max_length"] = max_steps
    CONFIG_DICT["wait_on_done"] = False # 回合结束后不自动重新开始。

env = tmrl.get_environment()
samples = []
output_path = None
all_finite = None
completed = False
benchmark = {}

with warnings.catch_warnings(record = True) as warning_records:
    warnings.simplefilter("always")
    try:
        actor = POLICY(env.observation_space , env.action_space).to_device(device)
        actor = actor.load(cfg.MODEL_PATH_WORKER , device = device)
        actor.eval()
        obs = tuple(np.zeros(space.shape , dtype = np.float32) for space in env.observation_space)
        processed_obs = OBS_PREPROCESSOR(obs)
        action = actor.act_(processed_obs , test = True)
        assert action.shape == (3,) and np.isfinite(action).all()
        assert env.action_space.contains(action)
        print("raw observation shapes:" , [part.shape for part in obs])
        print("policy observation shapes:" , [part.shape for part in processed_obs])
        print("synthetic action:" , action)
        print("Synthetic check passed. This does not verify game input, telemetry or driving.")

        if run_game:
            results_dir = project_root / "results/stage0"
            results_dir.mkdir(parents = True , exist_ok = True)
            output_path = results_dir / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jsonl")
            print("Real game check: one episode, at most" , max_steps , "steps.")
            started_at = time.time()
            obs , info = env.reset()
            all_finite = all(np.isfinite(part).all() for part in obs)
            assert all_finite , "Non-finite initial observation."
            # 在发送策略动作前，检查对应接口的视觉观测是否有效。
            if cfg.RTGYM_INTERFACE == "TM20LIDAR":
                assert np.any(obs[1] > 0) , "All initial LIDAR distances are zero; check the camera and captured game window."
            else:
                initial_image = obs[3][-1]
                assert obs[3].shape == (4 , 64 , 64) , "Unexpected image history shape."
                assert 0 <= initial_image.min() and initial_image.max() <= 255 , "Image pixels must be in [0, 255]."
                print("initial image min/max/std:" , float(initial_image.min()) , float(initial_image.max()) , float(initial_image.std()))
                assert np.ptp(initial_image) > 0 , "The initial captured image is uniform; check the captured game window."
            for step in range(max_steps):
                step_started = time.perf_counter()
                timestamp = time.time()
                action = actor.act_(OBS_PREPROCESSOR(obs) , test = True)
                all_finite = bool(np.isfinite(action).all())
                assert all_finite and env.action_space.contains(action) , "Invalid action."
                # RTGym 返回存在延迟的反馈，观测中保留两次历史动作。
                next_obs , reward , terminated , truncated , info = env.step(action)
                elapsed = time.perf_counter() - step_started
                all_finite = bool(np.isfinite(reward)) and all(np.isfinite(part).all() for part in next_obs)
                assert all_finite , "Non-finite observation or reward."
                samples.append({
                    "type": "transition" , "episode": 0 , "step": step ,
                    "timestamp": timestamp , "returned_at": time.time() , "elapsed_seconds": elapsed ,
                    "obs": [part.tolist() for part in obs] , "action": action.tolist() ,
                    "reward": float(reward) , "next_obs": [part.tolist() for part in next_obs] ,
                    "terminated": bool(terminated) , "truncated": bool(truncated)
                })
                obs = next_obs
                if terminated or truncated:
                    break
            completed = True
    finally:
        original_error = sys.exception()
        try:
            try:
                env.unwrapped.stop() # 先等待待执行动作结束，再释放手柄。
            finally:
                try:
                    if run_game and env.unwrapped.interface.j is not None:
                        env.unwrapped.interface.j.reset()
                        env.unwrapped.interface.j.update()
                        # 在解释器关闭前释放手柄，避免退出时才执行析构。
                        env.unwrapped.interface.j = None
                finally:
                    env.close()
            if output_path is not None:
                if samples:
                    for name , values in env.unwrapped.benchmarks().items():
                        benchmark[name] = [float(value) if value is not None and np.isfinite(value) else None for value in values]
                timeout_count = sum("Time-step timed out" in str(item.message) for item in warning_records)
                summary = {
                    "type": "summary" , "real_game": True , "completed": completed ,
                    "interface": cfg.RTGYM_INTERFACE , "run_name": cfg.RUN_NAME ,
                    "error": repr(original_error) if original_error is not None else None ,
                    "started_at": started_at , "ended_at": time.time() , "steps": len(samples) ,
                    "all_finite": all_finite , "time_step_timeouts": timeout_count ,
                    "observation_shapes": [list(space.shape) for space in env.observation_space] ,
                    "action_shape": list(env.action_space.shape) , "checkpoint": cfg.MODEL_PATH_WORKER ,
                    "reward_sum": sum(sample["reward"] for sample in samples) ,
                    "max_observed_speed": max((sample["next_obs"][0][0] for sample in samples) , default = None) ,
                    "mean_step_seconds": float(np.mean([sample["elapsed_seconds"] for sample in samples])) if samples else None ,
                    "benchmark": benchmark
                }
                with output_path.open("x" , encoding = "utf-8") as file:
                    for sample in samples:
                        file.write(json.dumps(sample , allow_nan = False) + "\n")
                    file.write(json.dumps(summary , allow_nan = False) + "\n")
                print("real game summary:" , summary)
                print("saved:" , output_path)
        except Exception as error:
            if original_error is None:
                raise
            print("Additional cleanup or recording error:" , error)
