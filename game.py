import ctypes
import sys
import time

import numpy as np
import torch

from replay_buffer import batch_observations


class GameCollector:
    def __init__(self , episode_steps , device , seed , startup_delay , total_steps = 0 , wait_on_done = True):
        # 构造时只保存参数，首次采集才初始化游戏接口
        self.episode_steps = episode_steps
        self.device = torch.device(device)
        self.seed = seed
        self.startup_delay = startup_delay
        self.total_steps = total_steps
        self.wait_on_done = wait_on_done
        self.env = None
        self.win32gui = None

    def _initialize(self , actor):
        # 必须先设置 DPI，再导入 TMRL，避免 Windows 缩放造成截图黑边
        if sys.platform == "win32":
            dpi_enabled = ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
            assert dpi_enabled , "无法启用 DPI 感知，请用新的 Python 进程运行。"

        import win32gui
        import tmrl.config.config_constants as cfg
        from tmrl.config.config_objects import CONFIG_DICT
        from tmrl.envs import GenericGymEnv

        self.win32gui = win32gui
        assert cfg.RTGYM_INTERFACE == "TM20IMAGES" and cfg.GRAYSCALE
        assert (cfg.IMG_HIST_LEN , cfg.IMG_WIDTH , cfg.IMG_HEIGHT , cfg.ACT_BUF_LEN) == (4 , 64 , 64 , 2)
        assert (cfg.WINDOW_WIDTH , cfg.WINDOW_HEIGHT) == (1280 , 640) and cfg.PRAGMA_GAMEPAD

        # 回合结束时由 RTGym 等待；此时才允许外部训练循环更新网络
        env_config = CONFIG_DICT.copy()
        env_config["ep_max_length"] = self.episode_steps
        env_config["wait_on_done"] = self.wait_on_done
        env_config["reset_act_buf"] = True
        env_config["benchmark"] = True
        self.env = GenericGymEnv(id = cfg.RTGYM_VERSION , gym_kwargs = {"config": env_config})
        self.env.action_space.seed(self.seed)

        # 提前运行策略，让首次 CUDA 初始化发生在采集开始之前
        actor.eval()
        initial_observation = tuple(np.zeros(space.shape , dtype = np.float32) for space in self.env.observation_space)
        with torch.no_grad():
            actor.sample(*batch_observations([initial_observation] , self.device))
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        print("请在" , self.startup_delay , "秒内切回 tmrl-test 游戏窗口；按 Esc 或 Ctrl+C 停止。")
        time.sleep(self.startup_delay)

    def check_stop(self):
        # 采集或训练期间按 Esc，交给外层统一清理并保存检查点
        if ctypes.windll.user32.GetAsyncKeyState(0x1B) & 0x8000:
            raise KeyboardInterrupt

    def _check_window(self , game_window):
        # 只接受前台普通窗口，避免窗口切换或最大化污染训练画面
        if not game_window or self.win32gui.GetForegroundWindow() != game_window:
            raise RuntimeError("Trackmania 不在前台，已停止采集，当前数据不进入回放池。")
        if ctypes.windll.user32.IsZoomed(game_window) or ctypes.windll.user32.IsIconic(game_window):
            raise RuntimeError("请使用 1280×640 普通窗口，不要最大化或最小化游戏。")

    def collect_episode(self , actor , buffer , warmup_steps , test = False):
        # 每次完整采集一个回合；游戏运行期间不执行梯度更新
        if self.env is None:
            self._initialize(actor)
        self.check_stop()
        game_window = self.win32gui.FindWindow(None , "Trackmania")
        self._check_window(game_window)

        actor.eval()
        observation , info = self.env.reset()
        self._check_window(game_window)
        assert observation[3].shape == (4 , 64 , 64) and np.ptp(observation[3][-1]) > 0
        episode_reward = 0.0
        episode_length = 0
        started = time.perf_counter()

        for step in range(self.episode_steps):
            self.check_stop()
            self._check_window(game_window)

            # 训练初期均匀探索；测试时直接使用 Actor 的确定性动作
            if not test and self.total_steps < warmup_steps:
                action = self.env.action_space.sample()
            else:
                with torch.no_grad():
                    action_tensor , log_prob = actor.sample(*batch_observations([observation] , self.device) , test = test)
                    action = action_tensor[0].cpu().numpy()
            assert np.isfinite(action).all() and self.env.action_space.contains(action)

            next_observation , reward , terminated , truncated , info = self.env.step(action)
            self._check_window(game_window)
            assert np.isfinite(reward) and all(np.isfinite(part).all() for part in next_observation)

            # 回放池保留原始观测；成功保存后立即计数，便于中断时保存进度
            buffer.add(observation , action , reward , next_observation , terminated , truncated)
            self.total_steps += 1
            observation = next_observation
            episode_reward += float(reward)
            episode_length += 1
            if terminated or truncated:
                break

        # 训练在 done 时等待并重置；测试保留结束画面，由外层立即释放控制
        assert terminated or truncated , "回合上限必须由 RTGym 返回 truncated，不能直接接着训练。"
        return {
            "steps": episode_length , "reward": episode_reward ,
            "terminated": bool(terminated) , "truncated": bool(truncated) ,
            "collection_seconds": time.perf_counter() - started
        }

    def close(self):
        # 仅在整个训练结束时释放控制，不能在回合间销毁手柄
        if self.env is None:
            return
        try:
            try:
                self.env.unwrapped.stop()
            finally:
                try:
                    if self.env.unwrapped.interface.j is not None:
                        self.env.unwrapped.interface.j.reset()
                        self.env.unwrapped.interface.j.update()
                        self.env.unwrapped.interface.j = None
                finally:
                    self.env.close()
        finally:
            self.env = None

