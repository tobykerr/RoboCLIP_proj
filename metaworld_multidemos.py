from gym import Env, spaces
import numpy as np
from stable_baselines3 import PPO
# from d4rl_alt.kitchen.kitchen_envs import KitchenMicrowaveHingeSlideV0, KitchenKettleV0
import torch as th
from s3dg import S3D
from gym.wrappers.time_limit import TimeLimit
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from PIL import Image, ImageSequence
import torch as th
from s3dg import S3D
import numpy as np
from PIL import Image, ImageSequence
import cv2
import gif2numpy
import PIL
import os
import seaborn as sns
import matplotlib.pylab as plt
from math import sqrt, ceil

from typing import Any, Dict

import gym
from gym.spaces import Box
import torch as th

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video
import os
from stable_baselines3.common.monitor import Monitor
from memory_profiler import profile
import argparse
from stable_baselines3.common.callbacks import EvalCallback

import metaworld
from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
                            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)

from kitchen_env_wrappers import readGif
from matplotlib import animation
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel

from metaworld_envs import MetaworldDense

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
    Builds a [num_demo, D] tensor of demo embeddings and saves it to out_path.
    This runs ONCE in the main process, so SubprocVecEnv workers don't redo it.
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
            x = int(center[1] - w/2)
            y = int(center[0] - h/2)
            frames = np.array([f[y:y+h, x:x+w] for f in frames])
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
            out = net(video)
            z = out["video_embedding"].detach().cpu()     # [1,D]
            embeds.append(z)

    demo_mat = th.cat(embeds, dim=0)  # [N,D]
    th.save(
        {
            "demo_embeddings": demo_mat,  # CPU tensor
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


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--algo', type=str, default='ppo')
    parser.add_argument('--text-string', type=str, default='robot opening sliding door')
    parser.add_argument('--dir-add', type=str, default='')
    parser.add_argument('--env-id', type=str, default='AssaultBullet-v0')
    parser.add_argument('--env-type', type=str, default='AssaultBullet-v0')
    parser.add_argument('--total-time-steps', type=int, default=5000000)
    parser.add_argument('--n-envs', type=int, default=8)
    parser.add_argument('--n-steps', type=int, default=128)
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument("--demo-dir", type=str, default="./gifs/custom/drawer-open-human")
    parser.add_argument("--num-demo", type=int, default=28)
    parser.add_argument("--demo-cache", type=str, default="demo_embeds.pt",
                    help="Path to saved demo embeddings .pt")
    parser.add_argument("--build-demo-cache", action="store_true",
                        help="If set, build demo cache then exit.")


    args = parser.parse_args()
    return args


class MetaworldSparseMultiBase(Env):
    def __init__(self, env_id, text_string=None, time=False, video_path=None, rank=0, human=True, num_demo=28, demo_cache_path="demo_embeds.pt"):
        super(MetaworldSparseMultiBase,self)
        self.num_demo = num_demo
        self.demo_cache_path = demo_cache_path
        door_open_goal_hidden_cls = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN[env_id]
        env = door_open_goal_hidden_cls(seed=rank)
        self.env = TimeLimit(env, max_episode_steps=128)
        self.time = time
        if not self.time:
            self.observation_space = self.env.observation_space
        else:
            self.observation_space = Box(low=-8.0, high=8.0, shape=(self.env.observation_space.shape[0]+1,), dtype=np.float32)
        self.action_space = self.env.action_space
        self.past_observations = []
        self.window_length = 16
        self.net = S3D('s3d_dict.npy', 512)

        # Load the model weights
        self.net.load_state_dict(th.load('s3d_howto100m.pth'))
        # Evaluation mode
        self.net = self.net.eval()
        self.targets = []
        # if video_path:
        #     for i in range(num_demo):
        #         path = video_path+f"/{i+1}.gif" # assume gifs placed in folder specified by video_demo and are named 1.gif, 2.gif, ...
        #         frames = readGif(path)
        #         # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$", len(frames))
        #         # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$", frames[0].shape)
        #         if human:
        #             frames = self.preprocess_human_demo(frames)
        #         else:
        #             frames = self.preprocess_metaworld(frames)
        #         if frames.shape[1]>3:
        #             frames = frames[:,:3]
        #         # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$", frames.shape)
        #         video = th.from_numpy(frames)
        #         video_output = self.net(video.float())
        #         self.targets.append(video_output['video_embedding'])

        # assert len(self.targets) == self.num_demo, f"Loaded {len(self.targets)} demos, expected {self.num_demo}"
        
        # load cache instead of reading gifs:
        cache = th.load(self.demo_cache_path, map_location="cpu")
        self.targets_mat = cache["demo_embeddings"]          # [N,D] CPU tensor
        assert self.targets_mat.shape[0] == self.num_demo, f"Cache has {self.targets_mat.shape[0]} demos, expected {self.num_demo}"
        self.targets = [self.targets_mat[i:i+1] for i in range(self.num_demo)]

        if rank == 0:
            print(f"[Init] env_id={env_id} | num_demo={self.num_demo} | loaded={len(self.targets)}")
            print(f"[Init] obs_space={self.observation_space} | act_space={self.action_space}")
            print(f"[Init] video_path={video_path} | human={human}")

        self.counter = 0

    def get_obs(self):
        return self.baseEnv._get_obs(self.baseEnv.prev_time_step)

    def preprocess_human_demo(self, frames):
        frames = np.array(frames)
        frames = frames[None, :,:,:,:]
        frames = frames.transpose(0, 4, 1, 2, 3)
        return frames

    def preprocess_metaworld(self, frames, shorten=True, crop=True):
        frames = np.array(frames)
        if crop:
            center = 240, 320
            h, w = (250, 250)
            x = int(center[1] - w/2)
            y = int(center[0] - h/2)
            # frames = np.array([cv2.resize(frame, dsize=(250, 250), interpolation=cv2.INTER_CUBIC) for frame in frames])
            frames = np.array([frame[y:y+h, x:x+w] for frame in frames])
        a = frames
        frames = frames[None, :,:,:,:]
        frames = frames.transpose(0, 4, 1, 2, 3)
        if shorten:
            frames = frames[:, :,::4,:,:]
        # frames = frames/255
        return frames
        
    
    def render(self):
        frame = self.env.render()
        # center = 240, 320
        # h, w = (250, 250)
        # x = int(center[1] - w/2)
        # y = int(center[0] - h/2)
        # frame = frame[y:y+h, x:x+w]
        return frame


    def step(self, action):
        """Take a step in environment, collect frames, and compute reward at the end.
        Note: reward computation is to be implemented in subclasses.
        Note: if compute_reward should take more than just video_embedding, override step() in subclass accordingly."""
        out = self.env.step(action)
        if len(out) == 4:
            obs, env_rew, done, info = out # robust to gym envs, got in a mess with this when setting up so adding this for safety
        else:
            obs, env_rew, terminated, truncated, info = out
            done = terminated or truncated

        #self.past_observations.append(self.env.render()) # original, below is more efficient (only keeps every 4th frame since this would happen in preprocess_metaworld anyway)
        frame = self.env.render()
        if frame is not None:
            #crop to 250x250 as in preprocess_metaworld
            center = 240, 320
            h, w = (250, 250)
            x = int(center[1] - w/2)
            y = int(center[0] - h/2)
            frame = frame[y:y+h, x:x+w]
            #only keep every 4th frame
            if self.counter % 4 == 0:
                self.past_observations.append(frame)

        self.counter += 1
        t = self.counter/128
        if self.time:
            obs = np.concatenate([obs, np.array([t])])
        if done:
            frames = self.preprocess_metaworld(self.past_observations, shorten=False, crop=False) # set shorten=False since we already did frame skipping above
            
        
        
            video = th.from_numpy(frames)
            # print("video.shape", video.shape)
            # print(frames.shape)
            video_output = self.net(video.float())

            video_embedding = video_output['video_embedding']

            reward = self.compute_reward(video_embedding)
            # reward = 0.0
            # for i in range(self.num_demo):
            #     similarity_matrix = th.matmul(self.targets[i], video_embedding.t())
            #     reward += similarity_matrix.detach().numpy()[0][0]

            return obs, reward, done, info
        return obs, 0.0, done, info
    
    def compute_reward(self, video_embedding):
        """Compute reward given video embedding. To be implemented in subclasses."""
        raise NotImplementedError

    def reset(self):
        self.past_observations = []
        self.counter = 0
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        if not self.time:
            return obs
        return np.concatenate([obs, np.array([0.0])])


class MetaworldSparseMultiMean(MetaworldSparseMultiBase):
    def __init__(self, env_id, text_string=None, time=False, video_path=None, rank=0, human=True, num_demo=28, demo_cache_path="demo_embeds.pt"):
        super().__init__(env_id, text_string, time, video_path, rank, human, num_demo, demo_cache_path)

    def compute_reward(self, video_embedding):
        """Sum the similarities from all demonstrations. Original approach used in the paper."""
        summed_rewards = 0.0
        for i in range(self.num_demo):
            similarity_matrix = th.matmul(self.targets[i], video_embedding.t())
            summed_rewards += similarity_matrix.detach().numpy()[0][0]
        reward = summed_rewards / self.num_demo
        return reward

class MetaworldSparseMultiMax(MetaworldSparseMultiBase):
    def __init__(self, env_id, text_string=None, time=False, video_path=None, rank=0, human=True, num_demo=28, demo_cache_path="demo_embeds.pt"):
        super().__init__(env_id, text_string, time, video_path, rank, human, num_demo, demo_cache_path)
    
    def compute_reward(self, video_embedding):
        """Take the maximum similarity from all demonstrations."""
        max_reward = -float('inf')
        for i in range(self.num_demo):
            similarity_matrix = th.matmul(self.targets[i], video_embedding.t())
            reward = similarity_matrix.detach().numpy()[0][0]
            if reward > max_reward:
                max_reward = reward
        return max_reward


class MetaworldSparseMultiMOM(MetaworldSparseMultiBase):
    """
    Median-of-means over *similarities* (scalar rewards), not embeddings.

    Reward at episode end:
      sims_i = <target_i, video_embedding>
      split sims into K groups, take mean per group, reward = median(group_means)
    """
    def __init__(
        self,
        env_id,
        text_string=None,
        time=False,
        video_path=None,
        rank=0,
        human=True,
        num_demo=28,
        mom_groups=None,          # optional override for K
        mom_seed=None,            # optional seed for grouping reproducibility
        demo_cache_path="demo_embeds.pt"
    ):
        super().__init__(env_id, text_string, time, video_path, rank, human, num_demo, demo_cache_path)

        # Sanity check
        assert len(self.targets) == self.num_demo, (
            f"Loaded {len(self.targets)} demos, expected {self.num_demo}. "
            f"Check video_path and filenames 1.gif..{self.num_demo}.gif"
        )

        # Stack demo embeddings once: [N, D]
        # (S3D returns [1, D] per demo; cat gives [N, D])
        self.targets_mat = th.cat(self.targets, dim=0).detach().cpu()

        self.N = self.targets_mat.shape[0]
        self.D = self.targets_mat.shape[1]

        # Choose number of groups K ~ sqrt(N)
        if mom_groups is None:
            self.K = max(1, int(sqrt(self.N)))
        else:
            self.K = int(mom_groups)
            self.K = max(1, min(self.K, self.N))

        # Make fixed random partition of indices into K groups
        g = th.Generator(device="cpu")
        if mom_seed is None:
            # If you want reproducible partitions per-rank, use a deterministic seed like (12345 + rank)
            mom_seed = 12345 + int(rank)
        g.manual_seed(int(mom_seed))

        perm = th.randperm(self.N, generator=g)
        self.groups = th.tensor_split(perm, self.K)  # list of index tensors

    def compute_reward(self, video_embedding: th.Tensor) -> float:
        """
        video_embedding: [1, D] (torch tensor)
        returns: scalar float reward = MoM(similarities)
        """
        # Ensure targets are on same device/dtype as video_embedding
        targets = self.targets_mat.to(device=video_embedding.device, dtype=video_embedding.dtype)

        # Similarities for all demos: [N]
        # targets: [N, D], video_embedding.t(): [D, 1] -> [N, 1] -> [N]
        sims = (targets @ video_embedding.t()).squeeze(-1)

        # Mean within each group, then median across groups
        group_means = th.stack([sims[idx].mean() for idx in self.groups], dim=0)  # [K]
        mom_sim = group_means.median()  # scalar tensor

        return float(mom_sim.detach().cpu())

# class MetaworldSparseMultiSequential(MetaworldSparseMultiBase):
#     def __init__(self, env_id, text_string=None, time=False, video_path=None, rank=0, human=True, num_demo=28, episodes_per_demo=1000):
#         super().__init__(env_id, text_string, time, video_path, rank, human, num_demo)
#         self.episodes_per_demo = episodes_per_demo
#         self.current_demo_index = 0
#         self.episodes_run_for_current_demo = 0
    
#     def compute_reward(self, video_embedding):
#         """Compute reward using sequential approach."""
#         similarity_matrix = th.matmul(self.targets[self.current_demo_index], video_embedding.t())
#         reward = similarity_matrix.detach().numpy()[0][0]
#         return reward
    
#     def step(self, action):
#         """Take a step in environment, collect frames, and compute reward at the end.
#         Note: reward computation is to be implemented in subclasses.
#         Note: if compute_reward should take more than just video_embedding, override step() in subclass accordingly."""
#         obs, _, done, info = self.env.step(action)
#         self.past_observations.append(self.env.render())

#         if self.episodes_run_for_current_demo >= self.episodes_per_demo:
#             self.current_demo_index = min(self.current_demo_index + 1, self.num_demo - 1) # move to next demo, but don't exceed bounds. NOTE: last demo will be used for remaining episodes if total episodes exceed num_demo * episodes_per_demo
#             self.episodes_run_for_current_demo = 0

#         self.episodes_run_for_current_demo += 1
#         self.counter += 1

#         t = self.counter/128

#         if self.time:
#             obs = np.concatenate([obs, np.array([t])])
#         if done:
#             frames = self.preprocess_metaworld(self.past_observations)
            
        
        
#             video = th.from_numpy(frames)
#             # print("video.shape", video.shape)
#             # print(frames.shape)
#             video_output = self.net(video.float())

#             video_embedding = video_output['video_embedding']

#             reward = self.compute_reward(video_embedding)

#             return obs, reward, done, info
#         return obs, 0.0, done, info


class MetaworldSparseMultiSequentialFixedBudget(MetaworldSparseMultiBase):
    """
    Use demo 0 for E episodes, then demo 1 for E episodes, etc.
    Reward is exactly the RoboCLIP single-demo dot product against the current demo.
    """

    def __init__(
        self,
        env_id,
        text_string=None,
        time=False,
        video_path=None,
        rank=0,
        human=True,
        num_demo=28,
        total_training_episodes=7813,   # rough estimate based on 1M steps / 128 steps per episode
        demo_cache_path="demo_embeds.pt"
    ):
        super().__init__(env_id, text_string, time, video_path, rank, human, num_demo, demo_cache_path)

        assert len(self.targets) == self.num_demo, (
            f"Loaded {len(self.targets)} demos, expected {self.num_demo}."
        )

        self.total_training_episodes = int(total_training_episodes)
        self.episodes_per_demo = int(ceil(self.total_training_episodes / self.num_demo))

        self.current_demo_index = 0
        self.episodes_used_on_current_demo = 0

    def compute_reward(self, video_embedding: th.Tensor) -> float:
        similarity_matrix = th.matmul(self.targets[self.current_demo_index], video_embedding.t())
        return float(similarity_matrix.detach().cpu().numpy()[0][0])

    def step(self, action):
        out = self.env.step(action)
        if len(out) == 4:
            obs, _, done, info = out
        else:
            obs, _, terminated, truncated, info = out
            done = terminated or truncated

        #self.past_observations.append(self.env.render()) # original, below is more efficient (only keeps every 4th frame since this would happen in preprocess_metaworld anyway)
        frame = self.env.render()
        if frame is not None:
            #crop to 250x250 as in preprocess_metaworld
            center = 240, 320
            h, w = (250, 250)
            x = int(center[1] - w/2)
            y = int(center[0] - h/2)
            frame = frame[y:y+h, x:x+w]
            #only keep every 4th frame
            if self.counter % 4 == 0:
                self.past_observations.append(frame)

        self.counter += 1
        t = self.counter / 128.0
        if self.time:
            obs = np.concatenate([obs, np.array([t], dtype=np.float32)])

        if done:
            frames = self.preprocess_metaworld(self.past_observations, shorten=False, crop=False)  # already did frame skipping and cropping above
            video = th.from_numpy(frames)

            with th.no_grad():
                video_output = self.net(video.float())
            video_embedding = video_output["video_embedding"]

            reward = self.compute_reward(video_embedding)

            # --- advance demo *per episode* ---
            self.episodes_used_on_current_demo += 1
            if self.episodes_used_on_current_demo >= self.episodes_per_demo:
                self.current_demo_index = min(self.current_demo_index + 1, self.num_demo - 1)
                self.episodes_used_on_current_demo = 0

            return obs, reward, done, info

        return obs, 0.0, done, info

    def reset(self):
        self.past_observations = []
        self.counter = 0
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        if not self.time:
            return obs
        return np.concatenate([obs, np.array([0.0])])

from collections import deque
import numpy as np
import torch as th

class MetaworldSparseMultiCentroidPrototype(MetaworldSparseMultiBase):
    """
    Prototype / centroid approach:
      proto = mean_i z_demo_i   (optionally normalizing)
      reward = <proto, z_rollout>   (or cosine if normalize_proto + normalize_rollout)

    Note: If you do not normalize anything, this is basically like "sum"
    but collapsed into a single vector. If you normalize, it behaves differently.
    With normalize_proto + normalize_rollout, it's cosine similarity to centroid.
    """
    def __init__(
        self,
        env_id,
        text_string=None,
        time=False,
        video_path=None,
        rank=0,
        human=True,
        num_demo=28,
        normalize_demos=False,
        normalize_proto=True,
        normalize_rollout=True,
        eps=1e-8,
        demo_cache_path="demo_embeds.pt"
    ):
        super().__init__(env_id, text_string, time, video_path, rank, human, num_demo, demo_cache_path)

        self.normalize_demos = bool(normalize_demos)
        self.normalize_proto = bool(normalize_proto)
        self.normalize_rollout = bool(normalize_rollout)
        self.eps = float(eps)

        # Stack demo embeddings: [N, D]
        demos = th.cat(self.targets, dim=0).detach().cpu()  # [N,D]

        if self.normalize_demos:
            demos = demos / (demos.norm(dim=1, keepdim=True) + self.eps)

        proto = demos.mean(dim=0, keepdim=True)  # [1,D]

        if self.normalize_proto:
            proto = proto / (proto.norm(dim=1, keepdim=True) + self.eps)

        self.prototype = proto  # CPU tensor [1,D]

    def compute_reward(self, video_embedding: th.Tensor) -> float:
        proto = self.prototype.to(device=video_embedding.device, dtype=video_embedding.dtype)  # [1,D]

        v = video_embedding
        if self.normalize_rollout:
            v = v / (v.norm(dim=1, keepdim=True) + self.eps)

        similarity_matrix = th.matmul(proto, v.t())  # [1,1]
        reward = similarity_matrix.squeeze()

        return float(reward.detach().cpu())

class MetaworldSparseMultiSoftmaxSim(MetaworldSparseMultiBase):
    """
    Softmax attention over similarities (scalar attention).

    sims_i = <z_demo_i, z_rollout>
    w = softmax(sims / tau)
    reward = sum_i w_i * sims_i

    This is "soft" max: tau -> 0 approaches max; tau -> inf approaches mean(sim).
    """
    def __init__(
        self,
        env_id,
        text_string=None,
        time=False,
        video_path=None,
        rank=0,
        human=True,
        num_demo=28,
        temperature=1.0,
        center_sims=True,
        demo_cache_path="demo_embeds.pt"
    ):
        super().__init__(env_id, text_string, time, video_path, rank, human, num_demo, demo_cache_path)
        self.temperature = float(temperature)
        assert self.temperature > 0, "temperature must be > 0"
        self.center_sims = bool(center_sims)

        # Stack demo embeddings once: [N, D]
        self.targets_mat = th.cat(self.targets, dim=0).detach().cpu()  # CPU is fine; moved to device per call

    def compute_reward(self, video_embedding: th.Tensor) -> float:
        # targets: [N, D] on same device/dtype
        targets = self.targets_mat.to(device=video_embedding.device, dtype=video_embedding.dtype)

        # sims: [N]
        sims = (targets @ video_embedding.t()).squeeze(-1)

        # stable softmax
        logits = sims / self.temperature
        if self.center_sims:
            logits = logits - logits.max()

        w = th.softmax(logits, dim=0)          # [N]
        reward = (w * sims).sum()              # scalar

        return float(reward.detach().cpu())

class MetaworldSparseMultiMostCentralDemo(MetaworldSparseMultiBase):
    """
    Pick the most central demo (a medoid) among the set, then run single-demo reward.
    Useful as baseline? (i.e. "best possible single demo approach")

    Two metrics:
      - metric="l2": pick i minimizing sum_j ||z_i - z_j||^2
      - metric="cosine": pick i maximizing mean cosine similarity to others

    Reward:
      reward = <z_demo_medoid, z_rollout>  (exact RoboCLIP matmul form)
    """
    def __init__(
        self,
        env_id,
        text_string=None,
        time=False,
        video_path=None,
        rank=0,
        human=True,
        num_demo=28,
        metric="cosine",
        eps=1e-8,
        demo_cache_path="demo_embeds.pt"
    ):
        super().__init__(env_id, text_string, time, video_path, rank, human, num_demo, demo_cache_path)

        self.metric = str(metric).lower()
        assert self.metric in ("l2", "cosine")
        self.eps = float(eps)

        # Stack demos: [N,D]
        demos = th.cat(self.targets, dim=0).detach().cpu()  # [N,D]
        N, D = demos.shape

        if self.metric == "l2":
            # Compute pairwise squared distances efficiently:
            # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
            sq = (demos * demos).sum(dim=1, keepdim=True)     # [N,1]
            gram = demos @ demos.t()                          # [N,N]
            d2 = sq + sq.t() - 2.0 * gram                    # [N,N]
            # numerical guard
            d2 = th.clamp(d2, min=0.0)
            score = d2.sum(dim=1)                             # [N] lower is better
            medoid_idx = int(th.argmin(score).item())

        else:  # cosine
            demos_n = demos / (demos.norm(dim=1, keepdim=True) + self.eps)
            sim = demos_n @ demos_n.t()                       # [N,N]
            # exclude self-similarity if you want; doesn't matter much but cleaner:
            sim = sim - th.eye(N)
            score = sim.mean(dim=1)                           # [N] higher is better
            medoid_idx = int(th.argmax(score).item())

        self.medoid_idx = medoid_idx
        self.medoid_embedding = demos[medoid_idx:medoid_idx+1]  # [1,D]

        print(f"[MostCentralDemo] metric={self.metric} selected demo index {self.medoid_idx} (0-based index)")

    def compute_reward(self, video_embedding: th.Tensor) -> float:
        target = self.medoid_embedding.to(device=video_embedding.device, dtype=video_embedding.dtype)
        similarity_matrix = th.matmul(target, video_embedding.t())
        return float(similarity_matrix.detach().cpu().numpy()[0][0])


def make_env(env_type, env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        # env = KitchenMicrowaveHingeSlideV0()
        if env_type == "mean":
            env = MetaworldSparseMultiMean(env_id=env_id, 
                                           video_path="./gifs/custom/drawer-open-human", 
                                           time=True, 
                                           rank=rank, 
                                           human=True,
                                           demo_cache_path=args.demo_cache,
                                           ) # FOR VIDEO REWARD, set human=False for metaworld demo
        elif env_type == "max":
            env = MetaworldSparseMultiMax(env_id=env_id, 
                                          video_path="./gifs/custom/drawer-open-human", 
                                          time=True, 
                                          rank=rank, 
                                          human=True,
                                          demo_cache_path=args.demo_cache,
                                          ) # FOR VIDEO REWARD, set human=False for metaworld demo
        elif env_type == "mom":
            env = MetaworldSparseMultiMOM(env_id=env_id, 
                                          video_path="./gifs/custom/drawer-open-human", 
                                          time=True, 
                                          rank=rank, 
                                          human=True,
                                          demo_cache_path=args.demo_cache,
                                          ) # FOR VIDEO REWARD, set human=False for metaworld demo
        # elif env_type == "sequential":
        #     env = MetaworldSparseMultiSequential(env_id=env_id, video_path="./gifs/human_opening_door", time=True, rank=rank, human=True, num_demo=4, episodes_per_demo=250) # FOR VIDEO REWARD, set human=False for metaworld demo
        # elif env_type == "sparse_original":
        #     env = KitchenEnvSparseOriginalReward(time=True)
        elif env_type == "softmax_sim":
            env = MetaworldSparseMultiSoftmaxSim(
                env_id=env_id,
                video_path="./gifs/custom/drawer-open-human",
                time=True,
                rank=rank,
                human=True,
                num_demo=28,
                temperature=1.0,
                demo_cache_path=args.demo_cache,
            )

        elif env_type == "centroid":
            env = MetaworldSparseMultiCentroidPrototype(
                env_id=env_id,
                video_path="./gifs/custom/drawer-open-human",
                time=True,
                rank=rank,
                human=True,
                num_demo=28,
                normalize_demos=False,
                normalize_proto=True,
                normalize_rollout=True, # normalize all for cosine similarity to centroid
                demo_cache_path=args.demo_cache,
            )

        elif env_type == "central_demo":
            env = MetaworldSparseMultiMostCentralDemo(
                env_id=env_id,
                video_path="./gifs/custom/drawer-open-human",
                time=True,
                rank=rank,
                human=True,
                num_demo=28,
                metric="cosine",
                demo_cache_path=args.demo_cache,
            )

        elif env_type == "seq_fixed":
            env = MetaworldSparseMultiSequentialFixedBudget(
                env_id=env_id,
                video_path="./gifs/custom/drawer-open-human",
                time=True,
                rank=rank,
                human=True,
                num_demo=28,
                total_training_episodes=7813,  # ≈ 1M env steps / 128 steps per episode
                demo_cache_path=args.demo_cache,
            )

        else:
            env = MetaworldDense(env_id=env_id, time=True, rank=rank)
        env = Monitor(env, os.path.join(log_dir, str(rank)))
        # env.seed(seed + rank)
        return env
    # set_global_seeds(seed)
    return _init

def main():
    global args
    global log_dir
    args = get_args()
    human = True # USING HUMAN DEMONSTRATIONS, REMEMBER TO CHANGE IF USING METAWORLD DEMOS
    # build cache if missing OR if explicitly requested
    if args.build_demo_cache or (not os.path.exists(args.demo_cache)):
        build_demo_cache(
            demo_dir=args.demo_dir,
            num_demo=args.num_demo,
            human=human,
            s3d_dict_path="s3d_dict.npy",
            s3d_weights_path="s3d_howto100m.pth",
            device="cpu",              # RoboCLIP-faithful
            out_path=args.demo_cache,
        )
        if args.build_demo_cache:
            return
    #env_id = "drawer-open-v2-goal-hidden"
    log_dir = f"metaworld/{args.env_id}_{args.env_type}{args.dir_add}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    envs = SubprocVecEnv([make_env(args.env_type, args.env_id, i) for i in range(args.n_envs)])

    if not args.pretrained:
        model = PPO("MlpPolicy", envs, verbose=1, tensorboard_log=log_dir, n_steps=args.n_steps, batch_size=args.n_steps*args.n_envs, n_epochs=1, ent_coef=0.5)
    else:
        model = PPO.load(args.pretrained, env=envs, tensorboard_log=log_dir)

    eval_env = SubprocVecEnv([make_env("dense_original", args.env_id, i) for i in range(10, 10+args.n_envs)])#KitchenEnvDenseOriginalReward(time=True)
    # Use deterministic actions for evaluation
    eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir,
                                 log_path=log_dir, eval_freq=500,
                                 deterministic=True, render=False)

    model.learn(total_timesteps=int(args.total_time_steps), callback=eval_callback)
    model.save(f"{log_dir}/trained")



if __name__ == '__main__':
    main()