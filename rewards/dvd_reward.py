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
        # frames = episode_frames[-self.cfg.clip_len :]
        T = len(episode_frames) # uniform sampling rather than just the end frames
        if T >= self.cfg.clip_len:
            idx = np.linspace(0, T-1, self.cfg.clip_len).astype(int)
            frames = [episode_frames[i] for i in idx]
        else:
            frames = episode_frames + [episode_frames[-1]] * (self.cfg.clip_len - T)

        if len(frames) < self.cfg.clip_len:
            frames = frames + [frames[-1]] * (self.cfg.clip_len - len(frames))

        rollout_clip = self._frames_to_clip(frames).to(self.device)
        print("Processed clip checksum:", int(torch.sum(rollout_clip).item())) # debug
        # TEMP DEBUG: dump rollout frames once
        if not hasattr(self, "_dumped_rollout"):
            self._dumped_rollout = True
            # save the first, middle, last frame of the *selected* clip
            clip = rollout_clip[0]  # [T,3,H,W]
            T = clip.shape[0]
            for name, idx in [("first", 0), ("mid", T//2), ("last", T-1)]:
                img = clip[idx].detach().cpu()  # [3,H,W] normalized
                # unnormalize for viewing
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
                std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
                img = img * std + mean
                img = (img.clamp(0,1) * 255).byte().permute(1,2,0).numpy()
                cv2.imwrite(f"debug_rollout_{name}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


        roll_emb = self._encode_clip(rollout_clip)

        logits = self.sim_discriminator.forward(roll_emb, self.demo_emb)  # (B,2)
        print("roll_emb mean/std:", float(roll_emb.mean()), float(roll_emb.std()))
        print("roll_emb norm:", float(torch.norm(roll_emb, dim=-1).item()))
        print("logits:", logits.detach().cpu().numpy())
        print("prob_same:", torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy())

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
        T = len(frames)
        if T == 0:
            raise ValueError(f"No frames in gif: {gif_path}")

        if T >= self.cfg.clip_len:
            idx = np.linspace(0, T-1, self.cfg.clip_len).astype(int)
            frames = [frames[i] for i in idx]
        else:
            frames = frames + [frames[-1]] * (self.cfg.clip_len - T)

        return self._frames_to_clip(frames)

    def _frames_to_clip(self, frames):
        proc = [self._preprocess_frame(f) for f in frames]   # list of (H,W,3) uint8 RGB
        clip = np.stack(proc, axis=0)                        # [T,H,W,3]
        clip = torch.from_numpy(clip).float() / 255.0        # [T,H,W,3] in [0,1]
        clip = clip.permute(0, 3, 1, 2)                      # [T,3,H,W]
        clip = clip.unsqueeze(0)                             # [1,T,3,H,W]

        # ImageNet normalization (as in transforms_video.py)
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=clip.dtype).view(1, 1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], dtype=clip.dtype).view(1, 1, 3, 1, 1)
        clip = (clip - mean) / std

        return clip


    def _preprocess_frame(self, frame):
        # frame is RGB uint8
        h, w = frame.shape[:2]
        s = min(h, w)
        y0 = (h - s) // 2
        x0 = (w - s) // 2
        frame = frame[y0:y0+s, x0:x0+s]  # central square crop
        # frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        frame = cv2.resize(frame, (self.cfg.resize_w, self.cfg.resize_h), interpolation=cv2.INTER_AREA)
        if not hasattr(self, "_dump_count"):
            self._dump_count = 0
        if self._dump_count < 5:
            cv2.imwrite(f"debug_preproc_{self._dump_count}.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            self._dump_count += 1
        return frame

