# rewards/dvd_reward.py
import os, sys
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import cv2
import imageio.v2 as imageio


@dataclass
class DVDConfig:
    clip_len: int = 20
    resize_h: int = 84     # DVD example uses 84x84 clips in __main__
    resize_w: int = 84
    device: str = "cuda"
    # reward from logits:
    #   "logit_diff" = (logit_same - logit_diffTask)
    #   "logprob_same" = log softmax prob of "same"
    reward_mode: str = "logit_diff"
    reward_scale: float = 1.0


class _Args:
    """Minimal args container to satisfy DVD modules."""
    def __init__(self, hidden_size=512, similarity=1):
        self.hidden_size = hidden_size
        self.similarity = similarity


class DVDReward:
    """
    Inference-only reward using anniesch/dvd:
      - MultiColumn video encoder (Model from model3D_1)
      - SimilarityDiscriminator
    """

    def __init__(
        self,
        dvd_repo_root: str,
        sim_discriminator_ckpt: str,   # e.g. third_party/dvd/pretrained/dvd_human_tasks_6_robot_tasks_3.pth.tar
        video_encoder_ckpt: str,       # e.g. third_party/dvd/pretrained/video_encoder/model_best.pth.tar
        demo_gif_path: str,
        cfg: Optional[DVDConfig] = None,
    ):
        self.cfg = cfg or DVDConfig()
        self.device = torch.device(self.cfg.device if torch.cuda.is_available() else "cpu")

        dvd_repo_root = os.path.abspath(dvd_repo_root)
        if dvd_repo_root not in sys.path:
            sys.path.insert(0, dvd_repo_root)

        # Import DVD modules exactly from the repo
        from multi_column import MultiColumn, SimilarityDiscriminator
        from model3D_1 import Model  # this is referenced in multi_column.py __main__

        # Build minimal args
        args = _Args(hidden_size=512, similarity=1)

        # Construct networks
        # num_classes is used by Model in many implementations; DVD uses Something-Something (174 classes) commonly.
        num_classes = 174
        column_units = 512

        self.video_encoder = MultiColumn(
            args=args,
            num_classes=num_classes,
            conv_column=Model,
            column_units=column_units,
            clf_layers=None,
        ).to(self.device).eval()

        self.sim_discriminator = SimilarityDiscriminator(args).to(self.device).eval()

        # Load encoder weights (checkpoint['state_dict'])
        enc_ckpt = torch.load(video_encoder_ckpt, map_location="cpu")
        if "state_dict" not in enc_ckpt:
            raise ValueError(f"Expected 'state_dict' in encoder ckpt: {video_encoder_ckpt}")
        self.video_encoder.load_state_dict(enc_ckpt["state_dict"], strict=False)

        # Load discriminator weights (raw state_dict)
        sim_sd = torch.load(sim_discriminator_ckpt, map_location="cpu")
        self.sim_discriminator.load_state_dict(sim_sd, strict=False)

        # Cache demo clip embedding
        demo_clip = self._load_gif_clip(demo_gif_path).to(self.device)
        with torch.no_grad():
            self.demo_emb = self._encode_clip(demo_clip)

    @torch.no_grad()
    def terminal_reward(self, episode_frames: List[np.ndarray]) -> float:
        frames = episode_frames[-self.cfg.clip_len :]
        if len(frames) < self.cfg.clip_len:
            frames = frames + [frames[-1]] * (self.cfg.clip_len - len(frames))

        rollout_clip = self._frames_to_clip(frames).to(self.device)
        roll_emb = self._encode_clip(rollout_clip)

        logits = self.sim_discriminator.forward(roll_emb, self.demo_emb)  # (B,2)

        # Choose which class index corresponds to "same task"
        # In DVD training code, positives are typically labeled as class 1, negatives class 0 (common CE convention).
        # We'll assume class 1 = "same task".
        logit_diff = (logits[:, 1] - logits[:, 0])  # scalar per batch
        logprob_same = F.log_softmax(logits, dim=-1)[:, 1]

        if self.cfg.reward_mode == "logit_diff":
            r = logit_diff.item()
        elif self.cfg.reward_mode == "logprob_same":
            r = logprob_same.item()
        else:
            raise ValueError(f"Unknown reward_mode: {self.cfg.reward_mode}")

        return float(self.cfg.reward_scale * r)

    # ---------------- encoding / preprocessing ----------------

    def _encode_clip(self, clip_b_tchw: torch.Tensor) -> torch.Tensor:
        """
        Our clip tensor is [B, T, 3, H, W].
        DVD MultiColumn expects a list of tensors shaped [B, 3, T, H, W] (or H/W swapped but 84x84 makes it moot).
        """
        clip_bcthw = clip_b_tchw.permute(0, 2, 1, 3, 4).contiguous()
        emb = self.video_encoder.encode([clip_bcthw])  # (B,512)
        return emb

    def _load_gif_clip(self, gif_path: str) -> torch.Tensor:
        frames = imageio.mimread(gif_path)
        frames = [np.asarray(f)[..., :3].copy() for f in frames]
        if len(frames) >= self.cfg.clip_len:
            frames = frames[-self.cfg.clip_len :]
        else:
            frames = frames + [frames[-1]] * (self.cfg.clip_len - len(frames))
        return self._frames_to_clip(frames)

    def _frames_to_clip(self, frames: List[np.ndarray]) -> torch.Tensor:
        proc = [self._preprocess_frame(f) for f in frames]  # list of (H,W,3)
        clip = np.stack(proc, axis=0)  # [T,H,W,3]
        clip = torch.from_numpy(clip).permute(0, 3, 1, 2).float() / 255.0  # [T,3,H,W]
        return clip.unsqueeze(0)  # [1,T,3,H,W]

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        f = cv2.resize(frame, (self.cfg.resize_w, self.cfg.resize_h), interpolation=cv2.INTER_AREA)
        return f
