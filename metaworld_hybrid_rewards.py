from gym import Env
import numpy as np
import torch as th
import os
import argparse

import gym
from gym.spaces import Box
from gym.wrappers.time_limit import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from s3dg import S3D
import metaworld
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_HIDDEN

from kitchen_env_wrappers import readGif
from metaworld_envs import MetaworldDense


# The reward definitions are:

# MetaworldSparseMultiMaxWithNearBonus
# max_i <z_demo_i, z_rollout> + near_bonus * 1[ever_near_object]

# MetaworldSparseMultiMaxWithNearAndMoveBonus
# max_i <z_demo_i, z_rollout> + near_bonus * 1[ever_near_object] + move_bonus * 1[drawer_moved_at_all]

# MetaworldSparseDenseEventsOnly
# near_bonus * 1[ever_near_object] + move_bonus * 1[drawer_moved_at_all]

# -----------------------------
# Demo cache utility
# -----------------------------

def build_demo_cache(
    demo_dir: str,
    num_demo: int,
    human: bool,
    s3d_dict_path: str = "s3d_dict.npy",
    s3d_weights_path: str = "s3d_howto100m.pth",
    device: str = "cpu",
    out_path: str = "demo_embeds.pt",
):
    """
    Build and save [num_demo, D] S3D demo embeddings once.
    """
    assert os.path.isdir(demo_dir), f"demo_dir not found: {demo_dir}"

    dev = th.device(device)
    net = S3D(s3d_dict_path, 512)
    net.load_state_dict(th.load(s3d_weights_path, map_location="cpu"))
    net.eval()
    net.to(dev)

    def preprocess_human(frames):
        frames = np.array(frames)[None, ...]          # [1,T,H,W,C]
        frames = frames.transpose(0, 4, 1, 2, 3)      # [1,C,T,H,W]
        return frames

    def preprocess_metaworld(frames, shorten=True, crop=True):
        frames = np.array(frames)
        if crop:
            center = 240, 320
            h, w = (250, 250)
            x = int(center[1] - w / 2)
            y = int(center[0] - h / 2)
            frames = np.array([f[y:y + h, x:x + w] for f in frames])
        frames = frames[None, ...]                    # [1,T,H,W,C]
        frames = frames.transpose(0, 4, 1, 2, 3)      # [1,C,T,H,W]
        if shorten:
            frames = frames[:, :, ::4, :, :]
        return frames

    embeds = []
    with th.no_grad():
        for i in range(num_demo):
            path = os.path.join(demo_dir, f"{i+1}.gif")
            frames = readGif(path)
            frames = preprocess_human(frames) if human else preprocess_metaworld(frames)
            if frames.shape[1] > 3:
                frames = frames[:, :3]
            video = th.from_numpy(frames).float().to(dev)
            T = video.shape[2]
            if T % 2 == 1:
                video = video[:, :, :-1]
            out = net(video)
            z = out["video_embedding"].detach().cpu()
            embeds.append(z)

    demo_mat = th.cat(embeds, dim=0)
    th.save(
        {
            "demo_embeddings": demo_mat,
            "meta": {
                "demo_dir": demo_dir,
                "num_demo": num_demo,
                "human": human,
                "s3d_dict": s3d_dict_path,
                "s3d_weights": s3d_weights_path,
            },
        },
        out_path,
    )
    print(f"[DemoCache] Saved {demo_mat.shape} -> {out_path}")
    return out_path


# -----------------------------
# Arguments
# -----------------------------

def get_args():
    parser = argparse.ArgumentParser(description="MetaWorld drawer-open hybrid rewards")
    parser.add_argument("--env-id", type=str, default="drawer-open-v2-goal-hidden")
    parser.add_argument("--env-type", type=str, default="max_near")
    parser.add_argument("--total-time-steps", type=int, default=1000000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--dir-add", type=str, default="")

    parser.add_argument("--demo-dir", type=str, default="./gifs/custom/drawer-open-human")
    parser.add_argument("--num-demo", type=int, default=28)
    parser.add_argument("--demo-cache", type=str, default="demo_embeds.pt")
    parser.add_argument("--build-demo-cache", action="store_true")

    # weak event rewards
    parser.add_argument("--near-bonus", type=float, default=0.10)
    parser.add_argument("--move-bonus", type=float, default=0.15)
    parser.add_argument("--move-threshold", type=float, default=0.01)

    return parser.parse_args()


# -----------------------------
# Base class: RoboCLIP-style terminal reward
# -----------------------------

class MetaworldSparseMultiBase(Env):
    def __init__(
        self,
        env_id,
        text_string=None,
        time=False,
        video_path=None,
        rank=0,
        human=True,
        num_demo=28,
        demo_cache_path="demo_embeds.pt",
    ):
        super(MetaworldSparseMultiBase, self).__init__()
        self.num_demo = num_demo
        self.demo_cache_path = demo_cache_path

        env_cls = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN[env_id]
        env = env_cls(seed=rank)
        self.env = TimeLimit(env, max_episode_steps=128)
        self.time = time

        if not self.time:
            self.observation_space = self.env.observation_space
        else:
            self.observation_space = Box(
                low=-8.0,
                high=8.0,
                shape=(self.env.observation_space.shape[0] + 1,),
                dtype=np.float32,
            )
        self.action_space = self.env.action_space

        self.past_observations = []
        self.counter = 0

        self.net = S3D("s3d_dict.npy", 512)
        self.net.load_state_dict(th.load("s3d_howto100m.pth", map_location="cpu"))
        self.net = self.net.eval()

        cache = th.load(self.demo_cache_path, map_location="cpu")
        self.targets_mat = cache["demo_embeddings"]
        assert self.targets_mat.shape[0] == self.num_demo, (
            f"Cache has {self.targets_mat.shape[0]} demos, expected {self.num_demo}"
        )
        self.targets = [self.targets_mat[i:i+1] for i in range(self.num_demo)]

        if rank == 0:
            print(f"[Init] env_id={env_id} | num_demo={self.num_demo} | loaded={len(self.targets)}")
            print(f"[Init] obs_space={self.observation_space} | act_space={self.action_space}")
            print(f"[Init] video_path={video_path} | human={human}")

    def preprocess_metaworld(self, frames, shorten=True, crop=True):
        frames = np.array(frames)
        if crop:
            center = 240, 320
            h, w = (250, 250)
            x = int(center[1] - w / 2)
            y = int(center[0] - h / 2)
            frames = np.array([frame[y:y + h, x:x + w] for frame in frames])
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
        if isinstance(obs, tuple):
            obs = obs[0]
        self._reset_episode_stats()
        if not self.time:
            return obs
        return np.concatenate([obs, np.array([0.0], dtype=np.float32)])

    def _reset_episode_stats(self):
        self.initial_obj_to_target = None
        self.final_obj_to_target = None
        self.ever_near_object = False
        self.ever_moved = False
        self.ever_success = False

    def _update_dense_stats(self, info):
        if not isinstance(info, dict):
            return

        obj_to_target = info.get("obj_to_target", None)
        near_object = info.get("near_object", None)
        success = info.get("success", None)

        if obj_to_target is not None:
            obj_to_target = float(obj_to_target)
            if self.initial_obj_to_target is None:
                self.initial_obj_to_target = obj_to_target
            self.final_obj_to_target = obj_to_target

        if near_object is not None:
            self.ever_near_object = self.ever_near_object or bool(near_object)

        if success is not None:
            self.ever_success = self.ever_success or bool(success)

    def _capture_frame(self):
        frame = self.env.render()
        if frame is None:
            return
        center = 240, 320
        h, w = (250, 250)
        x = int(center[1] - w / 2)
        y = int(center[0] - h / 2)
        frame = frame[y:y + h, x:x + w]
        if self.counter % 4 == 0:
            self.past_observations.append(frame)

    def _compute_video_embedding(self):
        frames = self.preprocess_metaworld(self.past_observations, shorten=False, crop=False)
        video = th.from_numpy(frames).float()
        T = video.shape[2]
        if T % 2 == 1:
            video = video[:, :, :-1]
        with th.no_grad():
            video_output = self.net(video)
        return video_output["video_embedding"]

    def _compute_roboclip_max_reward(self, video_embedding: th.Tensor) -> float:
        max_reward = -float("inf")
        for i in range(self.num_demo):
            target = self.targets[i].to(device=video_embedding.device, dtype=video_embedding.dtype)
            sim = th.matmul(target, video_embedding.t())
            val = float(sim.detach().cpu().numpy()[0][0])
            if val > max_reward:
                max_reward = val
        return max_reward

    def _compute_dense_event_reward(self):
        raise NotImplementedError

    def _combine_rewards(self, video_embedding: th.Tensor) -> float:
        raise NotImplementedError

    def step(self, action):
        out = self.env.step(action)
        if len(out) == 4:
            obs, _, done, info = out
        else:
            obs, _, terminated, truncated, info = out
            done = terminated or truncated

        self._update_dense_stats(info)
        self._capture_frame()

        self.counter += 1
        if self.final_obj_to_target is not None and self.initial_obj_to_target is not None:
            if (self.initial_obj_to_target - self.final_obj_to_target) > getattr(self, "move_threshold", 0.01):
                self.ever_moved = True

        t = self.counter / 128.0
        if self.time:
            obs = np.concatenate([obs, np.array([t], dtype=np.float32)])

        if done:
            video_embedding = self._compute_video_embedding()
            reward = self._combine_rewards(video_embedding)
            return obs, reward, done, info

        return obs, 0.0, done, info


# -----------------------------
# Dense helper mixin behaviour
# -----------------------------

class _SimpleDenseEventReward:
    """
    Weak event-based shaping only.

    Reward:
        near_bonus * 1[ever_near_object]
      + move_bonus * 1[drawer_moved_at_all]

    This is intentionally too weak to solve drawer-open reliably on its own.
    """
    def __init__(self, near_bonus=0.10, move_bonus=0.15, move_threshold=0.01, **kwargs):
        self.near_bonus = float(near_bonus)
        self.move_bonus = float(move_bonus)
        self.move_threshold = float(move_threshold)
        super().__init__(**kwargs)

    def _compute_dense_event_reward(self):
        reward = 0.0
        reward += self.near_bonus * float(self.ever_near_object)
        reward += self.move_bonus * float(self.ever_moved)
        return reward


# -----------------------------
# Requested classes
# -----------------------------

class MetaworldSparseMultiMaxWithNearBonus(_SimpleDenseEventReward, MetaworldSparseMultiBase):
    """
    RoboCLIP max terminal reward + weak near-object bonus.

    Reward at episode end:
        max_i <z_demo_i, z_rollout> + near_bonus * 1[ever_near_object]
    """
    def _compute_dense_event_reward(self):
        return self.near_bonus * float(self.ever_near_object)

    def _combine_rewards(self, video_embedding: th.Tensor) -> float:
        return self._compute_roboclip_max_reward(video_embedding) + self._compute_dense_event_reward()


class MetaworldSparseMultiMaxWithNearAndMoveBonus(_SimpleDenseEventReward, MetaworldSparseMultiBase):
    """
    RoboCLIP max terminal reward + two weak dense terms.

    Reward at episode end:
        max_i <z_demo_i, z_rollout>
        + near_bonus * 1[ever_near_object]
        + move_bonus * 1[drawer_moved_at_all]
    """
    def _combine_rewards(self, video_embedding: th.Tensor) -> float:
        return self._compute_roboclip_max_reward(video_embedding) + self._compute_dense_event_reward()


class MetaworldSparseDenseEventsOnly(_SimpleDenseEventReward, MetaworldSparseMultiBase):
    """
    Dense terms only, with no RoboCLIP component.

    Reward at episode end:
        near_bonus * 1[ever_near_object]
      + move_bonus * 1[drawer_moved_at_all]

    This is useful as the control to check whether the weak event terms are
    insufficient on their own but useful when paired with RoboCLIP.
    """
    def _combine_rewards(self, video_embedding: th.Tensor) -> float:
        return self._compute_dense_event_reward()


# -----------------------------
# Env factory and training entrypoint
# -----------------------------

args = None
log_dir = None


def make_env(env_type, env_id, rank, num_demo, seed=0):
    def _init():
        common = dict(
            env_id=env_id,
            video_path=args.demo_dir,
            time=True,
            rank=rank,
            human=True,
            num_demo=num_demo,
            demo_cache_path=args.demo_cache,
        )

        if env_type == "max_near":
            env = MetaworldSparseMultiMaxWithNearBonus(
                near_bonus=args.near_bonus,
                move_bonus=args.move_bonus,
                move_threshold=args.move_threshold,
                **common,
            )
        elif env_type == "max_near_move":
            env = MetaworldSparseMultiMaxWithNearAndMoveBonus(
                near_bonus=args.near_bonus,
                move_bonus=args.move_bonus,
                move_threshold=args.move_threshold,
                **common,
            )
        elif env_type == "dense_events_only":
            env = MetaworldSparseDenseEventsOnly(
                near_bonus=args.near_bonus,
                move_bonus=args.move_bonus,
                move_threshold=args.move_threshold,
                **common,
            )
        else:
            env = MetaworldDense(env_id=env_id, time=True, rank=rank)

        env = Monitor(env, os.path.join(log_dir, str(rank)))
        return env

    return _init


def main():
    global args
    global log_dir
    args = get_args()

    human = True
    if args.build_demo_cache or (not os.path.exists(args.demo_cache)):
        build_demo_cache(
            demo_dir=args.demo_dir,
            num_demo=args.num_demo,
            human=human,
            s3d_dict_path="s3d_dict.npy",
            s3d_weights_path="s3d_howto100m.pth",
            device="cpu",
            out_path=args.demo_cache,
        )
        if args.build_demo_cache:
            return

    log_dir = f"metaworld/{args.env_id}_{args.env_type}{args.dir_add}"
    os.makedirs(log_dir, exist_ok=True)

    envs = SubprocVecEnv([
        make_env(env_type=args.env_type, env_id=args.env_id, rank=i, num_demo=args.num_demo)
        for i in range(args.n_envs)
    ])

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
        )
    else:
        model = PPO.load(args.pretrained, env=envs, tensorboard_log=log_dir)

    eval_env = SubprocVecEnv([
        make_env(env_type="dense_original", env_id=args.env_id, rank=i + 10, num_demo=args.num_demo)
        for i in range(args.n_envs)
    ])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=500,
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=int(args.total_time_steps), callback=eval_callback)
    model.save(f"{log_dir}/trained")


if __name__ == "__main__":
    main()

