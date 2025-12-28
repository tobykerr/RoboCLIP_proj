# debug_dvd_task_separation.py
#
# Tests whether the pretrained DVD model produces different "prob_same" (and reward)
# when comparing ONE fixed demo (e.g., a human drawer-open GIF) against rollouts from
# MULTIPLE MetaWorld tasks.
#
# Outputs:
#  - CSV with per-episode results
#  - Summary TXT
#  - Matplotlib plots saved into --log-dir
#
# Usage example:
#   python debug_dvd_task_separation.py \
#     --demo-gif ./gifs/drawer-open-human2.gif \
#     --tasks drawer-open-v2-goal-hidden drawer-close-v2-goal-hidden button-press-v2-goal-hidden \
#     --n-episodes 20 \
#     --log-dir logs/dvd_tasksep

import os
import csv
import argparse
from dataclasses import asdict
from typing import List, Dict, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from gym.wrappers.time_limit import TimeLimit
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_HIDDEN

from rewards.dvd_reward import DVDReward, DVDConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--demo-gif", type=str, required=True)
    p.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=[
            "drawer-open-v2-goal-hidden",
            "drawer-close-v2-goal-hidden",
            "button-press-v2-goal-hidden",
            "door-open-v2-goal-hidden",
            "reach-v2-goal-hidden",
        ],
    )
    p.add_argument("--n-episodes", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-dir", type=str, default="logs/dvd_tasksep")

    # DVD config bits you might want to vary quickly
    p.add_argument("--clip-len", type=int, default=20)
    p.add_argument("--reward-mode", type=str, default="logit_diff", choices=["logit_diff", "logprob_same"])
    p.add_argument("--reward-scale", type=float, default=1.0)
    p.add_argument("--resize", type=int, default=84, help="DVD frames are resized to resize x resize inside DVDReward")

    # paths to pretrained checkpoints (your defaults)
    p.add_argument("--dvd-repo-root", type=str, default="third_party/dvd")
    p.add_argument("--dvd-sim-ckpt", type=str, default="third_party/dvd/pretrained/dvd_human_tasks_6_robot_tasks_3.pth.tar")
    p.add_argument("--dvd-enc-ckpt", type=str, default="third_party/dvd/pretrained/video_encoder/model_best.pth.tar")

    return p.parse_args()


def make_env(env_id: str, seed: int, max_steps: int):
    env_cls = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN[env_id]
    env = env_cls(seed=seed)
    env = TimeLimit(env, max_episode_steps=max_steps)
    return env


def collect_random_episode(env, max_steps: int) -> Tuple[List[np.ndarray], Dict]:
    frames = []
    obs = env.reset()
    info = {}
    for _ in range(max_steps):
        frames.append(env.render())  # RGB uint8
        action = env.action_space.sample()
        obs, r, done, info = env.step(action)
        if done:
            frames.append(env.render())
            break
    return frames, info


@torch.no_grad()
def dvd_logits_prob_same_and_reward(dvd: DVDReward, frames: List[np.ndarray]) -> Tuple[np.ndarray, float, float]:
    """
    Computes:
      - logits from sim_discriminator(roll_emb, demo_emb)
      - prob_same = softmax(logits)[same_class]
      - reward = dvd.terminal_reward(frames) (your chosen reward_mode)
    """
    L = dvd.cfg.clip_len
    clip_frames = frames[-L:]
    if len(clip_frames) < L:
        clip_frames = clip_frames + [clip_frames[-1]] * (L - len(clip_frames))

    clip = dvd._frames_to_clip(clip_frames).to(dvd.device)   # [1,T,3,H,W] normalized
    roll_emb = dvd._encode_clip(clip)                        # [1,512]
    logits = dvd.sim_discriminator(roll_emb, dvd.demo_emb)   # [1,2]
    probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()

    # By your assumption in DVDReward: class 1 = "same task"
    prob_same = float(probs[1])

    # Reward (logit_diff or logprob_same)
    reward = float(dvd.terminal_reward(frames))

    return logits[0].detach().cpu().numpy(), prob_same, reward


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_histogram(values_by_task: Dict[str, List[float]], title: str, xlabel: str, out_path: str):
    plt.figure()
    # Overlayed histograms (matplotlib chooses default colors automatically)
    for task, vals in values_by_task.items():
        if len(vals) == 0:
            continue
        plt.hist(vals, bins=15, alpha=0.5, label=task)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_boxplot(values_by_task: Dict[str, List[float]], title: str, ylabel: str, out_path: str):
    plt.figure(figsize=(max(8, 1.2 * len(values_by_task)), 4))
    labels = list(values_by_task.keys())
    data = [values_by_task[k] for k in labels]
    plt.boxplot(data, labels=labels, showfliers=True)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    args = parse_args()
    ensure_dir(args.log_dir)

    # Build DVDReward once (fixed demo across tasks)
    dvd = DVDReward(
        dvd_repo_root=args.dvd_repo_root,
        sim_discriminator_ckpt=args.dvd_sim_ckpt,
        video_encoder_ckpt=args.dvd_enc_ckpt,
        demo_gif_path=args.demo_gif,
        cfg=DVDConfig(
            clip_len=args.clip_len,
            resize_h=args.resize,
            resize_w=args.resize,
            device="cuda",
            reward_mode=args.reward_mode,
            reward_scale=args.reward_scale,
        ),
    )

    # Log config
    cfg_path = os.path.join(args.log_dir, "config.txt")
    with open(cfg_path, "w") as f:
        f.write("ARGS:\n")
        for k, v in vars(args).items():
            f.write(f"  {k}: {v}\n")
        f.write("\nDVDConfig:\n")
        for k, v in asdict(dvd.cfg).items():
            f.write(f"  {k}: {v}\n")

    # Per-episode CSV
    csv_path = os.path.join(args.log_dir, "episodes.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "episode_idx", "success", "logit0", "logit1", "prob_same", "reward"])

    # Collect results
    prob_by_task: Dict[str, List[float]] = {t: [] for t in args.tasks}
    rew_by_task: Dict[str, List[float]] = {t: [] for t in args.tasks}
    succ_by_task: Dict[str, List[float]] = {t: [] for t in args.tasks}

    summary_lines = []
    summary_lines.append("Per-task results (mean ± std):\n")

    for ti, task in enumerate(args.tasks):
        env = make_env(task, seed=args.seed + ti * 1000, max_steps=args.max_steps)

        for ep in range(args.n_episodes):
            frames, info = collect_random_episode(env, max_steps=args.max_steps)

            logits, prob_same, reward = dvd_logits_prob_same_and_reward(dvd, frames)

            success = info.get("success", None)
            if success is None:
                # Some envs may not set it; keep as nan
                success = float("nan")
            else:
                success = float(success)

            prob_by_task[task].append(prob_same)
            rew_by_task[task].append(reward)
            succ_by_task[task].append(success)

            # append to CSV
            with open(csv_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([task, ep, success, float(logits[0]), float(logits[1]), prob_same, reward])

            print(f"[{task}] ep {ep:03d} | success={success} | prob_same={prob_same:.6f} | reward={reward:.6f} | logits={logits}")

    # Write summaries
    for task in args.tasks:
        p = np.array(prob_by_task[task], dtype=np.float64)
        r = np.array(rew_by_task[task], dtype=np.float64)
        s = np.array(succ_by_task[task], dtype=np.float64)

        summary_lines.append(
            f"{task}\n"
            f"  prob_same: mean={np.nanmean(p):.6f}, std={np.nanstd(p):.6f}, min={np.nanmin(p):.6f}, max={np.nanmax(p):.6f}\n"
            f"  reward   : mean={np.nanmean(r):.6f}, std={np.nanstd(r):.6f}, min={np.nanmin(r):.6f}, max={np.nanmax(r):.6f}\n"
            f"  success  : mean={np.nanmean(s):.6f} (nan means env didn't report success)\n"
        )

    summary_path = os.path.join(args.log_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))

    # Plots
    save_histogram(
        prob_by_task,
        title="DVD prob_same distributions by task (fixed demo)",
        xlabel="prob_same",
        out_path=os.path.join(args.log_dir, "prob_same_hist.png"),
    )
    save_boxplot(
        prob_by_task,
        title="DVD prob_same by task (boxplot)",
        ylabel="prob_same",
        out_path=os.path.join(args.log_dir, "prob_same_box.png"),
    )
    save_histogram(
        rew_by_task,
        title=f"DVD reward distributions by task (mode={args.reward_mode}, fixed demo)",
        xlabel="reward",
        out_path=os.path.join(args.log_dir, "reward_hist.png"),
    )
    save_boxplot(
        rew_by_task,
        title="DVD reward by task (boxplot)",
        ylabel="reward",
        out_path=os.path.join(args.log_dir, "reward_box.png"),
    )

    # Also plot mean±std bars (prob_same)
    plt.figure(figsize=(max(8, 1.2 * len(args.tasks)), 4))
    means = [np.mean(prob_by_task[t]) for t in args.tasks]
    stds = [np.std(prob_by_task[t]) for t in args.tasks]
    x = np.arange(len(args.tasks))
    plt.bar(x, means, yerr=stds, capsize=3)
    plt.xticks(x, args.tasks, rotation=30, ha="right")
    plt.ylabel("prob_same (mean ± std)")
    plt.title("DVD prob_same mean±std by task (fixed demo)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.log_dir, "prob_same_mean_std.png"), dpi=150)
    plt.close()

    print(f"\nWrote:\n  {csv_path}\n  {summary_path}\n  plots into {args.log_dir}\n")


if __name__ == "__main__":
    main()
