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
from math import sqrt

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


    args = parser.parse_args()
    return args


class MetaworldSparseMultiBase(Env):
    def __init__(self, env_id, text_string=None, time=False, video_path=None, rank=0, human=True, num_demo=1):
        super(MetaworldSparseMultiBase,self)
        self.num_demo = num_demo
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
        # self.net.load_state_dict(th.load('s3d_howto100m.pth'))
        # Evaluation mode
        self.net = self.net.eval()
        self.targets = []
        if video_path:
            for i in range(num_demo):
                path = video_path+f"/button-press-v2_{i}.gif"
                frames = readGif(path)
                print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$", len(frames))
                print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$", frames[0].shape)
                if human:
                    frames = self.preprocess_human_demo(frames)
                else:
                    frames = self.preprocess_metaworld(frames)
                if frames.shape[1]>3:
                    frames = frames[:,:3]
                print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$", frames.shape)
                video = th.from_numpy(frames)
                video_output = self.net(video.float())
                self.targets.append(video_output['video_embedding'])

        self.counter = 0

    def get_obs(self):
        return self.baseEnv._get_obs(self.baseEnv.prev_time_step)

    def preprocess_human_demo(self, frames):
        frames = np.array(frames)
        frames = frames[None, :,:,:,:]
        frames = frames.transpose(0, 4, 1, 2, 3)
        return frames

    def preprocess_metaworld(self, frames, shorten=True):
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
        obs, _, done, info = self.env.step(action)
        self.past_observations.append(self.env.render())
        self.counter += 1
        t = self.counter/128
        if self.time:
            obs = np.concatenate([obs, np.array([t])])
        if done:
            frames = self.preprocess_metaworld(self.past_observations)
            
        
        
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
        if not self.time:
            return self.env.reset()
        return np.concatenate([self.env.reset(), np.array([0.0])])


class MetaworldSparseMultiOriginalSum(MetaworldSparseMultiBase):
    def __init__(self, env_id, text_string=None, time=False, video_path=None, rank=0, human=True, num_demo=1):
        super().__init__(env_id, text_string, time, video_path, rank, human, num_demo)
    
    def compute_reward(self, video_embedding):
        """Sum the similarities from all demonstrations. Original approach used in the paper."""
        reward = 0.0
        for i in range(self.num_demo):
            similarity_matrix = th.matmul(self.targets[i], video_embedding.t())
            reward += similarity_matrix.detach().numpy()[0][0]
        return reward

class MetaworldSparseMultiMax(MetaworldSparseMultiBase):
    def __init__(self, env_id, text_string=None, time=False, video_path=None, rank=0, human=True, num_demo=1):
        super().__init__(env_id, text_string, time, video_path, rank, human, num_demo)
    
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
    def __init__(self, env_id, text_string=None, time=False, video_path=None, rank=0, human=True, num_demo=1):
        super().__init__(env_id, text_string, time, video_path, rank, human, num_demo)
        self.mom_embedding = self.find_mom_embedding()

    def find_mom_embedding(self):
        """Splits demos into K groups, computes mean of each group, and returns median of these means."""
        N = self.num_demo
        K = int(sqrt(N))
        np.random.shuffle(self.targets)  # shuffle in-place

        groups = np.array_split(self.targets, K) # split into K groups
        group_means = [np.mean(group, axis=0) for group in groups] # find mean of each group
        
        mom_embedding = np.median(np.stack(group_means), axis=0) # compute element-wise median across group means
                                                                 # NOTE: should I use geomtric median? Will take longer (iteration needed) but is more robust.
        
        mom_embedding = th.tensor(mom_embedding) # NOTE: needed? Check later.
        return mom_embedding

    def compute_reward(self, video_embedding):
        """Compute reward using median-of-means embedding."""
        similarity_matrix = th.matmul(self.mom_embedding, video_embedding.t())
        reward = similarity_matrix.detach().numpy()[0][0]
        return reward

class MetaworldSparseMultiSequential(MetaworldSparseMultiBase):
    def __init__(self, env_id, text_string=None, time=False, video_path=None, rank=0, human=True, num_demo=1, episodes_per_demo=1000):
        super().__init__(env_id, text_string, time, video_path, rank, human, num_demo)
        self.episodes_per_demo = episodes_per_demo
        self.current_demo_index = 0
        self.episodes_run_for_current_demo = 0
    
    def compute_reward(self, video_embedding):
        """Compute reward using sequential approach."""
        similarity_matrix = th.matmul(self.targets[self.current_demo_index], video_embedding.t())
        reward = similarity_matrix.detach().numpy()[0][0]
        return reward
    
    def step(self, action):
        """Take a step in environment, collect frames, and compute reward at the end.
        Note: reward computation is to be implemented in subclasses.
        Note: if compute_reward should take more than just video_embedding, override step() in subclass accordingly."""
        obs, _, done, info = self.env.step(action)
        self.past_observations.append(self.env.render())

        if self.episodes_run_for_current_demo >= self.episodes_per_demo:
            self.current_demo_index = min(self.current_demo_index + 1, self.num_demo - 1) # move to next demo, but don't exceed bounds. NOTE: last demo will be used for remaining episodes if total episodes exceed num_demo * episodes_per_demo
            self.episodes_run_for_current_demo = 0

        self.episodes_run_for_current_demo += 1
        self.counter += 1

        t = self.counter/128

        if self.time:
            obs = np.concatenate([obs, np.array([t])])
        if done:
            frames = self.preprocess_metaworld(self.past_observations)
            
        
        
            video = th.from_numpy(frames)
            # print("video.shape", video.shape)
            # print(frames.shape)
            video_output = self.net(video.float())

            video_embedding = video_output['video_embedding']

            reward = self.compute_reward(video_embedding)

            return obs, reward, done, info
        return obs, 0.0, done, info