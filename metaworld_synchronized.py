# Global plateau-based demo switching (synchronized across ALL envs)
# - Reward computed exactly like RoboCLIP:
#     similarity_matrix = th.matmul(target_embedding, video_embedding.t())
#     reward = similarity_matrix.detach().numpy()[0][0]
# - SubprocVecEnv + PPO + EvalCallback (same overall experiment style)
#
# OOM fix (same as before):
# - Do NOT read/encode demo gifs inside each SubprocVecEnv worker.
# - Load precomputed demo embeddings from a single .pt cache file instead.
#
# Notes:
# - PPO can still run on GPU via --ppo-device cuda (this matches SB3 behavior),
#   but S3D inference remains CPU, as in RoboCLIP's released code.
# - Using torch.no_grad() would be a safe optimization, but RoboCLIP did not use it.
#   This script therefore does not use it, for faithfulness.

from __future__ import annotations

import os
import argparse
from collections import deque
from typing import Optional

import numpy as np
import torch as th
from gym import Env
from gym.spaces import Box
from gym.wrappers.time_limit import TimeLimit

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

import metaworld  # noqa: F401
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_HIDDEN

from s3dg import S3D


# -------------------------
# Args
# -------------------------
def get_args():
    p = argparse.ArgumentParser(
        description="RoboCLIP multi-demo (SubprocVecEnv + global plateau switching) [faithful S3D=CPU]"
    )

    # RoboCLIP-style
    p.add_argument("--env-id", type=str, default="drawer-open-v2-goal-hidden")
    p.add_argument("--env-type", type=str, default="sparse_multidemo_plateau")  # routes make_env
    p.add_argument("--dir-add", type=str, default="")
    p.add_argument("--total-time-steps", type=int, default=1_000_000)  # RoboCLIP README: stopped at 1M env steps
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--n-steps", type=int, default=128)
    p.add_argument("--pretrained", type=str, default=None)

    # Demos
    p.add_argument("--demo-dir", type=str, default="./gifs/custom/drawer-open-human")
    p.add_argument("--num-demo", type=int, default=28)

    # >>> OOM FIX: load demos from a precomputed cache .pt <<<
    p.add_argument(
        "--demo-cache",
        type=str,
        default="demo_embeds.pt",
        help="Path to precomputed demo embeddings .pt file (contains demo_embeddings: [N,D]).",
    )

    # Control whether demo preprocessing is "human" vs "metaworld" (same meaning as RoboCLIP)
    # Python 3.9+: BooleanOptionalAction gives --human / --no-human
    try:
        p.add_argument("--human", action=argparse.BooleanOptionalAction, default=True)
    except AttributeError:
        # Fallback for older Python: default True; use --no-human to disable
        p.add_argument("--human", action="store_true", default=True)
        p.add_argument("--no-human", dest="human", action="store_false")

    # Convenience alias: if set, force human=False (metaworld-style preprocessing)
    p.add_argument(
        "--metaworld-demo",
        action="store_true",
        default=False,
        help="If set, force human=False (use metaworld-style preprocessing).",
    )

    # S3D (faithful: CPU)
    p.add_argument("--s3d-dict", type=str, default="s3d_dict.npy")
    p.add_argument("--s3d-weights", type=str, default="s3d_howto100m.pth")

    # PPO device (policy/value nets)
    p.add_argument(
        "--ppo-device",
        type=str,
        default="cuda" if th.cuda.is_available() else "cpu",
        help="Device for PPO policy/value net: cuda or cpu.",
    )

    # Plateau switching (global controller)
    p.add_argument("--window-episodes", type=int, default=50,
                   help="Rolling window size in completed episodes (GLOBAL across all envs).")
    p.add_argument("--patience-windows", type=int, default=6,
                   help="How many consecutive windows without improvement before switching.")
    p.add_argument("--min-delta", type=float, default=0.01,
                   help="Minimum moving-average improvement to reset plateau counter.")
    p.add_argument("--drop-delta", type=float, default=0.05,
                   help="If MA drops by >= drop-delta from best MA, switch immediately.")
    p.add_argument("--warmup-episodes", type=int, default=100,
                   help="Don’t switch before this many completed episodes on current demo (GLOBAL).")
    p.add_argument("--max-episodes-per-demo", type=int, default=3000,
                   help="Hard cap (GLOBAL completed episodes) before forced switch.")
    p.add_argument("--switch-verbose", type=int, default=1,
                   help="1 = print on demo switch; 0 = silent.")

    return p.parse_args()


# -------------------------
# Sparse env: multi-demo but single-demo reward (exact RoboCLIP matmul)
# -------------------------
class MetaworldSparseMultiSynchronized(Env):
    """
    Like RoboCLIP MetaworldSparse, but:
    - loads multiple demo embeddings (precomputed .pt cache)
    - reward uses ONLY current demo embedding (exact same matmul form as RoboCLIP)
    - current demo index is set externally (synchronized via callback)
    - S3D stays on CPU (faithful to RoboCLIP code)
    """

    def __init__(
        self,
        env_id: str,
        time: bool = True,
        demo_dir: Optional[str] = None,
        num_demo: int = 28,
        rank: int = 0,
        human: bool = True,
        s3d_dict_path: str = "s3d_dict.npy",
        s3d_weights_path: str = "s3d_howto100m.pth",
        max_episode_steps: int = 128,
        # >>> OOM FIX: cache path <<<
        demo_cache_path: str = "demo_embeds.pt",
    ):
        super().__init__()

        self.env_id = env_id
        self.num_demo = int(num_demo)
        self.demo_dir = demo_dir  # kept for compatibility / logging, but not used for loading now
        self.current_demo_index = 0
        self.demo_cache_path = demo_cache_path

        env_cls = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN[self.env_id]
        base_env = env_cls(seed=rank)
        self.env = TimeLimit(base_env, max_episode_steps=max_episode_steps)

        self.time = bool(time)
        if not self.time:
            self.observation_space = self.env.observation_space
        else:
            self.observation_space = Box(
                low=-8.0, high=8.0,
                shape=(self.env.observation_space.shape[0] + 1,),
                dtype=np.float32,
            )
        self.action_space = self.env.action_space

        self.max_episode_steps = int(max_episode_steps)
        self.counter = 0
        self.past_observations = []

        # ---- S3D (faithful: CPU) ----
        self.net = S3D(s3d_dict_path, 512)
        self.net.load_state_dict(th.load(s3d_weights_path, map_location="cpu"))
        self.net = self.net.eval()

        # >>> OOM FIX: Load demo embeddings from cache instead of reading gifs in each worker <<<
        cache = th.load(self.demo_cache_path, map_location="cpu")
        demo_mat = cache["demo_embeddings"] if isinstance(cache, dict) and "demo_embeddings" in cache else cache
        assert isinstance(demo_mat, th.Tensor), "demo_cache must contain a torch Tensor or dict with key 'demo_embeddings'"
        assert demo_mat.dim() == 2, f"Expected demo_embeddings to have shape [N,D], got {tuple(demo_mat.shape)}"
        assert demo_mat.shape[0] == self.num_demo, (
            f"Cache has {demo_mat.shape[0]} demos, expected {self.num_demo}. "
            f"Check --num-demo and the cache file."
        )

        # Store as list of [1,D] tensors, matching previous code paths
        self.targets = [demo_mat[i:i + 1].detach().cpu() for i in range(self.num_demo)]

    # Called by callback via VecEnv.env_method (works with SubprocVecEnv)
    def set_current_demo_index(self, idx: int):
        idx = int(idx)
        idx = max(0, min(idx, self.num_demo - 1))
        self.current_demo_index = idx

    def preprocess_human_demo(self, frames):
        frames = np.array(frames)             # [T,H,W,C]
        frames = frames[None, :, :, :, :]     # [1,T,H,W,C]
        frames = frames.transpose(0, 4, 1, 2, 3)  # [1,C,T,H,W]
        return frames

    def preprocess_metaworld(self, frames, shorten=True):
        center = 240, 320
        h, w = (250, 250)
        x = int(center[1] - w / 2)
        y = int(center[0] - h / 2)
        frames = np.array([frame[y:y + h, x:x + w] for frame in frames])  # [T,250,250,C]
        frames = frames[None, :, :, :, :]
        frames = frames.transpose(0, 4, 1, 2, 3)
        if shorten:
            frames = frames[:, :, ::4, :, :]
        return frames

    def render(self):
        return self.env.render()

    def reset(self):
        self.past_observations = []
        self.counter = 0
        obs = self.env.reset()
        # gymnasium compat
        if isinstance(obs, tuple) and len(obs) == 2:
            obs, _info = obs
        if not self.time:
            return obs
        return np.concatenate([obs, np.array([0.0], dtype=np.float32)])

    def step(self, action):
        out = self.env.step(action)
        if len(out) == 4:
            obs, _, done, info = out
        else:
            obs, _, terminated, truncated, info = out
            done = terminated or truncated

        frame = self.env.render()
        if frame is not None:
            self.past_observations.append(frame)

        self.counter += 1
        t = self.counter / float(self.max_episode_steps)
        if self.time:
            obs = np.concatenate([obs, np.array([t], dtype=np.float32)])

        if done:
            frames = self.preprocess_metaworld(self.past_observations)
            video = th.from_numpy(frames)  # CPU
            video_output = self.net(video.float())  # faithful: no no_grad()
            video_embedding = video_output["video_embedding"].detach().cpu()  # [1,D] CPU

            # ---- EXACT RoboCLIP single-demo reward computation ----
            target_embedding = self.targets[self.current_demo_index]              # [1,D] CPU
            similarity_matrix = th.matmul(target_embedding, video_embedding.t())  # [1,1]
            reward = similarity_matrix.detach().numpy()[0][0]  # EXACT form

            info = dict(info)
            info["roboclip_reward"] = float(reward)
            info["demo_index"] = int(self.current_demo_index)

            return obs, float(reward), done, info

        return obs, 0.0, done, info


# -------------------------
# Dense eval env (same as RoboCLIP MetaworldDense)
# -------------------------
class MetaworldDense(Env):
    def __init__(self, env_id: str, time: bool = True, rank: int = 0, max_episode_steps: int = 128):
        super().__init__()
        env_cls = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN[env_id]
        base_env = env_cls(seed=rank)
        self.env = TimeLimit(base_env, max_episode_steps=max_episode_steps)

        self.time = bool(time)
        if not self.time:
            self.observation_space = self.env.observation_space
        else:
            self.observation_space = Box(
                low=-8.0, high=8.0,
                shape=(self.env.observation_space.shape[0] + 1,),
                dtype=np.float32,
            )
        self.action_space = self.env.action_space

        self.counter = 0
        self.max_episode_steps = int(max_episode_steps)

    def render(self):
        return self.env.render()

    def reset(self):
        self.counter = 0
        obs = self.env.reset()
        if isinstance(obs, tuple) and len(obs) == 2:
            obs, _info = obs
        if not self.time:
            return obs
        return np.concatenate([obs, np.array([0.0], dtype=np.float32)])

    def step(self, action):
        out = self.env.step(action)
        if len(out) == 4:
            obs, reward, done, info = out
        else:
            obs, reward, terminated, truncated, info = out
            done = terminated or truncated

        self.counter += 1
        t = self.counter / float(self.max_episode_steps)
        if self.time:
            obs = np.concatenate([obs, np.array([t], dtype=np.float32)])

        return obs, float(reward), done, info


# -------------------------
# Global plateau-based switching callback (synchronized across SubprocVecEnv)
# -------------------------
class GlobalPlateauDemoSwitchCallback(BaseCallback):
    """
    Switches the demo index for ALL envs together when reward plateaus/drops.

    Reads per-episode reward from info["roboclip_reward"] on done=True.
    Uses VecEnv.env_method to broadcast to SubprocVecEnv workers.
    """

    def __init__(
        self,
        num_demo: int,
        window_episodes: int = 50,
        patience_windows: int = 6,
        min_delta: float = 0.01,
        drop_delta: float = 0.05,
        warmup_episodes: int = 100,
        max_episodes_per_demo: int = 3000,
        switch_verbose: int = 1,
        verbose: int = 0,
    ):
        super().__init__(verbose)

        self.num_demo = int(num_demo)
        self.window_episodes = int(window_episodes)
        self.patience_windows = int(patience_windows)
        self.min_delta = float(min_delta)
        self.drop_delta = float(drop_delta)
        self.warmup_episodes = int(warmup_episodes)
        self.max_episodes_per_demo = int(max_episodes_per_demo)
        self.switch_verbose = int(switch_verbose)

        self.current_demo = 0
        self.recent_rewards = deque(maxlen=self.window_episodes)
        self.best_ma = -float("inf")
        self.no_improve_windows = 0
        self.episodes_on_demo = 0

        # bookkeeping
        self.episodes_per_demo = [0 for _ in range(self.num_demo)]
        self.used_demos = set()

    def _broadcast_demo(self, new_demo: int):
        new_demo = max(0, min(int(new_demo), self.num_demo - 1))
        self.current_demo = new_demo

        # SubprocVecEnv-safe broadcast
        self.training_env.env_method("set_current_demo_index", self.current_demo)

        # reset stats for this demo
        self.recent_rewards.clear()
        self.best_ma = -float("inf")
        self.no_improve_windows = 0
        self.episodes_on_demo = 0

        if self.switch_verbose:
            print(f"[GlobalPlateauDemoSwitch] Switched to demo {self.current_demo + 1}/{self.num_demo}")

    def _on_training_start(self) -> None:
        self._broadcast_demo(0)

    def _on_training_end(self) -> None:
        used = sorted(self.used_demos)

        print("\n========== DEMO USAGE SUMMARY ==========")
        print(f"Total demos available: {self.num_demo}")
        print(f"Demos actually used : {len(used)}")
        for i in used:
            print(f"  Demo {i+1:02d}: {self.episodes_per_demo[i]} episodes")
        print("========================================\n")

        # write to disk (inside tensorboard log dir)
        log_dir = self.logger.dir
        if log_dir:
            path = os.path.join(log_dir, "demo_usage.txt")
            with open(path, "w") as f:
                f.write("Demo usage summary\n")
                f.write(f"Total demos available: {self.num_demo}\n")
                f.write(f"Demos actually used: {len(used)}\n\n")
                for i in used:
                    f.write(f"Demo {i+1}: {self.episodes_per_demo[i]} episodes\n")

    def _maybe_switch(self):
        if self.current_demo >= self.num_demo - 1:
            return
        if self.episodes_on_demo < self.warmup_episodes:
            return
        if len(self.recent_rewards) < self.window_episodes:
            return

        ma = float(np.mean(self.recent_rewards))

        # light tensorboard logging (no spammy prints)
        self.logger.record("plateau/current_demo", self.current_demo)
        self.logger.record("plateau/ma_reward", ma)
        self.logger.record("plateau/best_ma", self.best_ma)
        self.logger.record("plateau/episodes_on_demo", self.episodes_on_demo)

        # improvement?
        if ma > self.best_ma + self.min_delta:
            self.best_ma = ma
            self.no_improve_windows = 0
            return

        # drop?
        if (self.best_ma - ma) >= self.drop_delta:
            self._broadcast_demo(self.current_demo + 1)
            return

        # plateau
        self.no_improve_windows += 1
        if self.no_improve_windows >= self.patience_windows:
            self._broadcast_demo(self.current_demo + 1)
            return

        # hard cap
        if self.episodes_on_demo >= self.max_episodes_per_demo:
            self._broadcast_demo(self.current_demo + 1)

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        if dones is None or infos is None:
            return True

        for d, info in zip(dones, infos):
            if not d or info is None:
                continue
            if "roboclip_reward" not in info:
                continue

            r = float(info["roboclip_reward"])
            self.recent_rewards.append(r)
            self.episodes_on_demo += 1
            self.episodes_per_demo[self.current_demo] += 1
            self.used_demos.add(self.current_demo)

        self._maybe_switch()
        return True


# -------------------------
# Env factory (RoboCLIP style)
# -------------------------
def make_env(args, env_type: str, env_id: str, rank: int, log_dir: str):
    def _init():
        if env_type == "sparse_multidemo_plateau":
            # force human False if --metaworld-demo
            human = bool(args.human) and (not bool(args.metaworld_demo))
            env = MetaworldSparseMultiSynchronized(
                env_id=env_id,
                time=True,
                demo_dir=args.demo_dir,
                num_demo=args.num_demo,
                rank=rank,
                human=human,
                s3d_dict_path=args.s3d_dict,
                s3d_weights_path=args.s3d_weights,
                max_episode_steps=128,
                # >>> OOM FIX: pass cache path into each worker <<<
                demo_cache_path=args.demo_cache,
            )
        else:
            env = MetaworldDense(env_id=env_id, time=True, rank=rank, max_episode_steps=128)

        env = Monitor(env, os.path.join(log_dir, str(rank)))
        return env

    return _init


def main():
    args = get_args()

    log_dir = f"metaworld/{args.env_id}_{args.env_type}{args.dir_add}"
    os.makedirs(log_dir, exist_ok=True)

    # training envs (SubprocVecEnv, RoboCLIP-style)
    envs = SubprocVecEnv([make_env(args, args.env_type, args.env_id, i, log_dir) for i in range(args.n_envs)])

    # model (PPO device can be cuda; S3D remains CPU)
    if not args.pretrained:
        model = PPO(
            "MlpPolicy",
            envs,
            verbose=1,
            tensorboard_log=log_dir,
            n_steps=args.n_steps,
            batch_size=args.n_steps * args.n_envs,
            n_epochs=1,
            ent_coef=0.5,
            device=args.ppo_device,
        )
    else:
        model = PPO.load(args.pretrained, env=envs, tensorboard_log=log_dir, device=args.ppo_device)

    # eval env (dense original reward, RoboCLIP-style)
    eval_env = SubprocVecEnv([make_env(args, "dense_original", args.env_id, 10 + i, log_dir) for i in range(args.n_envs)])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=500,
        deterministic=True,
        render=False,
    )

    # global demo scheduler callback
    demo_switch_callback = GlobalPlateauDemoSwitchCallback(
        num_demo=args.num_demo,
        window_episodes=args.window_episodes,
        patience_windows=args.patience_windows,
        min_delta=args.min_delta,
        drop_delta=args.drop_delta,
        warmup_episodes=args.warmup_episodes,
        max_episodes_per_demo=args.max_episodes_per_demo,
        switch_verbose=args.switch_verbose,
        verbose=0,
    )

    model.learn(total_timesteps=int(args.total_time_steps), callback=[demo_switch_callback, eval_callback])
    model.save(f"{log_dir}/trained")


if __name__ == "__main__":
    main()
