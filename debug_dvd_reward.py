import numpy as np
import torch
from gym.wrappers.time_limit import TimeLimit
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_HIDDEN

from rewards.dvd_reward import DVDReward, DVDConfig

ENV_ID = "drawer-open-v2-goal-hidden"
DEMO_GIF = "./gifs/drawer-open-human2.gif"

def collect_random_episode(env, max_steps=128):
    frames = []
    obs = env.reset()
    info = {}
    for _ in range(max_steps):
        frames.append(env.render())
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if done:
            frames.append(env.render())
            break
    return frames, info

def checksum(img):
    return int(np.sum(img))

def collect_episode(env, action_fn, max_steps=128):
    frames = []
    obs = env.reset()
    info = {}
    for t in range(max_steps):
        frames.append(env.render())
        a = action_fn(t, env)
        obs, r, done, info = env.step(a)
        if done:
            frames.append(env.render())
            break
    return frames, info

def main():
    env_cls = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN[ENV_ID]
    env = TimeLimit(env_cls(seed=0), max_episode_steps=128)

    dvd = DVDReward(
        dvd_repo_root="third_party/dvd",
        sim_discriminator_ckpt="third_party/dvd/pretrained/dvd_human_tasks_6_robot_tasks_3.pth.tar",
        video_encoder_ckpt="third_party/dvd/pretrained/video_encoder/model_best.pth.tar",
        demo_gif_path=DEMO_GIF,
        cfg=DVDConfig(
            clip_len=20,
            resize_h=84,
            resize_w=84,
            reward_mode="logit_diff",
            reward_scale=1.0,
        ),
    )

    # --- Thrash test (zero vs thrash) ---
    zero_fn = lambda t, env: np.zeros(env.action_space.shape, dtype=np.float32)

    def thrash_fn(t, env):
        a = env.action_space.sample().astype(np.float32)
        return np.clip(3.0 * a, env.action_space.low, env.action_space.high).astype(np.float32)

    frames0, _ = collect_episode(env, zero_fn)
    frames1, _ = collect_episode(env, thrash_fn)

    r0 = dvd.terminal_reward(frames0)
    r1 = dvd.terminal_reward(frames1)
    print("Reward (zero):", r0)
    print("Reward (thrash):", r1)

    # --- Random episodes sanity ---
    rewards = []
    for ep in range(3):
        frames, info = collect_random_episode(env)
        f0 = frames[0].astype(np.int32)
        fL = frames[-1].astype(np.int32)
        print("Mean abs pixel diff first-last:", np.mean(np.abs(fL - f0)))
        print("Checksums:", checksum(frames[0]), checksum(frames[len(frames)//2]), checksum(frames[-1]))

        r = dvd.terminal_reward(frames)
        rewards.append(r)
        print(f"Episode {ep:02d} | reward = {r:.4f} | success = {info.get('success', None)}")

    rewards = np.array(rewards)
    print("Reward mean:", rewards.mean())
    print("Reward std :", rewards.std())

    # Processed clip stats (for the last collected random episode)
    clip = dvd._frames_to_clip(frames[-dvd.cfg.clip_len:]).cpu().numpy()
    print("Processed clip checksum:", int(np.sum(clip)))
    print("Processed clip mean/std:", float(clip.mean()), float(clip.std()))

    # Demo-vs-demo sanity check
    demo_clip = dvd._load_gif_clip(DEMO_GIF).to(dvd.device)
    demo_emb = dvd._encode_clip(demo_clip)
    logits = dvd.sim_discriminator(demo_emb, demo_emb)
    print("Demo-demo logits:", logits.detach().cpu().numpy())
    print("Demo-demo softmax:", torch.softmax(logits, dim=-1).detach().cpu().numpy())

if __name__ == "__main__":
    main()
