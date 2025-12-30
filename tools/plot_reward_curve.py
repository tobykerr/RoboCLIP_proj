#!/usr/bin/env python3
import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EVAL_COLOR = "tab:blue"
ROBOCLIP_COLOR = "tab:orange"


def ensure_dir(d: str):
    Path(d).mkdir(parents=True, exist_ok=True)


def savefig(out_dir: str, name: str):
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, name)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def load_evaluations_npz(npz_path: str):
    data = np.load(npz_path)
    timesteps = data["timesteps"]          # [K]
    results = data["results"]              # [K, n_eval_episodes]
    return timesteps, results


def read_one_monitor_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#")    # Monitor header lines start with '#'
    if df.shape[0] == 0:
        return df
    df["source_file"] = os.path.basename(path)
    base = os.path.basename(path)
    try:
        rank = int(base.split(".")[0])
    except Exception:
        rank = None
    df["rank"] = rank
    return df


def load_monitor_dir_or_glob(monitor_glob: str) -> pd.DataFrame:
    paths = sorted(glob.glob(monitor_glob))
    if not paths:
        raise FileNotFoundError(f"No monitor CSVs matched: {monitor_glob}")
    dfs = []
    for p in paths:
        df = read_one_monitor_csv(p)
        if df.shape[0] > 0:
            dfs.append(df)
    if not dfs:
        raise RuntimeError(f"Monitor CSVs found but empty: {monitor_glob}")
    return pd.concat(dfs, ignore_index=True)


def split_train_vs_eval_monitors(mdf: pd.DataFrame, n_envs: int):
    """
    Your code uses:
      - train ranks: 0..n_envs-1
      - eval ranks:  10..10+n_envs-1
    """
    if "rank" not in mdf.columns or mdf["rank"].isna().all():
        raise RuntimeError("Could not parse rank from monitor filenames (expected '0.monitor.csv', etc).")

    train_ranks = set(range(0, n_envs))
    eval_ranks = set(range(10, 10 + n_envs))

    train_df = mdf[mdf["rank"].isin(train_ranks)].copy()
    eval_df = mdf[mdf["rank"].isin(eval_ranks)].copy()
    other_df = mdf[~mdf["rank"].isin(train_ranks | eval_ranks)].copy()

    return train_df, eval_df, other_df


def make_global_episode_index(df: pd.DataFrame) -> pd.DataFrame:
    # Sort within each worker by wall-clock t, then concatenate deterministically
    df = df.sort_values(["rank", "t"]).reset_index(drop=True)
    df["episode_idx_global"] = np.arange(len(df))
    return df


def add_approx_timesteps(df: pd.DataFrame) -> pd.DataFrame:
    if "l" not in df.columns:
        raise RuntimeError("Monitor DF missing column 'l' (episode length).")
    df = df.copy()
    df["approx_timesteps"] = df["l"].cumsum()
    return df


def plot_line(x, y, color, xlabel, ylabel, title, out_dir, fname):
    plt.figure()
    plt.plot(x, y, color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    return savefig(out_dir, fname)


def plot_line_rolling_mean_only(x, y, color, window, xlabel, ylabel, title, out_dir, fname):
    w = max(1, int(window))
    sm = pd.Series(y).rolling(w, min_periods=1).mean().to_numpy()
    plt.figure()
    plt.plot(x, sm, color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{title} (rolling mean, w={w})")
    return savefig(out_dir, fname)


def plot_eval_mean_with_std_band(timesteps, results, out_dir, fname, title_suffix=""):
    eval_mean = results.mean(axis=1)
    eval_std = results.std(axis=1)

    plt.figure()
    plt.plot(timesteps, eval_mean, color=EVAL_COLOR, label="mean")
    plt.fill_between(
        timesteps,
        eval_mean - eval_std,
        eval_mean + eval_std,
        color=EVAL_COLOR,
        alpha=0.2,
        label="±1 std"
    )
    plt.xlabel("Training timesteps")
    plt.ylabel("Eval reward (dense env reward)")
    plt.title(f"Dense eval reward (mean ± std){title_suffix}")
    plt.legend()
    return savefig(out_dir, fname)


def main():
    ap = argparse.ArgumentParser(
        description="Plot dense eval reward (evaluations.npz) + RoboCLIP training reward (monitor.csv), without mixing eval monitors."
    )
    ap.add_argument("--run-dir", required=True, type=str,
                    help="Directory containing evaluations.npz and monitor CSVs (your log_dir).")
    ap.add_argument("--out-dir", default="plots", type=str)
    ap.add_argument("--n-envs", default=8, type=int,
                    help="Number of training env workers (and eval env workers). Used to split monitor ranks.")
    ap.add_argument("--monitor-glob", default=None, type=str,
                    help="Glob for monitor files. Default: <run-dir>/*.monitor.csv")
    ap.add_argument("--eval-npz", default=None, type=str,
                    help="Path to evaluations.npz. Default: <run-dir>/evaluations.npz")

    # Rolling mean is only used for the *separate* rolling-mean-only plots you requested
    ap.add_argument("--train-rolling-window", default=50, type=int,
                    help="Rolling window for RoboCLIP rolling-mean-only plots (episodes).")

    ap.add_argument("--also-plot-eval-monitor-dense", action="store_true",
                    help="Optionally also plot dense reward from eval monitor CSVs (r) for sanity check.")
    ap.add_argument("--combined", action="store_true",
                    help="Also produce a combined plot: RoboCLIP reward (raw) vs timesteps (left axis) and eval mean dense reward (right axis).")

    args = ap.parse_args()

    run_dir = args.run_dir
    out_dir = args.out_dir
    ensure_dir(out_dir)

    monitor_glob = args.monitor_glob or os.path.join(run_dir, "*.monitor.csv")
    eval_npz = args.eval_npz or os.path.join(run_dir, "evaluations.npz")

    # ----------------------------
    # Eval dense rewards (no smoothing)
    # ----------------------------
    have_eval = os.path.exists(eval_npz)
    if not have_eval:
        print(f"[Warn] evaluations.npz not found at: {eval_npz}")
    else:
        timesteps, results = load_evaluations_npz(eval_npz)
        eval_mean = results.mean(axis=1)
        eval_std = results.std(axis=1)

        # Mean curve (raw)
        plot_line(
            timesteps, eval_mean, EVAL_COLOR,
            xlabel="Training timesteps",
            ylabel="Eval mean reward (dense env reward)",
            title="Dense eval reward (mean) vs timesteps",
            out_dir=out_dir,
            fname="eval_dense_mean_vs_timesteps.png"
        )

        # Mean ± std band (this is your “± std bars” request, but as a band—more readable)
        plot_eval_mean_with_std_band(
            timesteps, results,
            out_dir=out_dir,
            fname="eval_dense_mean_pm_std_vs_timesteps.png"
        )

        # Std curve (raw)
        plot_line(
            timesteps, eval_std, EVAL_COLOR,
            xlabel="Training timesteps",
            ylabel="Eval reward std (across eval episodes)",
            title="Dense eval reward std vs timesteps",
            out_dir=out_dir,
            fname="eval_dense_std_vs_timesteps.png"
        )

    # ----------------------------
    # Load monitor CSVs and split
    # ----------------------------
    mdf = load_monitor_dir_or_glob(monitor_glob)
    train_df, eval_df, other_df = split_train_vs_eval_monitors(mdf, n_envs=args.n_envs)

    if len(train_df) == 0:
        raise RuntimeError(f"No training monitor rows found. Expected ranks 0..{args.n_envs-1} in filenames.")

    train_df = make_global_episode_index(train_df)
    train_df = add_approx_timesteps(train_df)

    # ----------------------------
    # RoboCLIP plots: RAW and ROLLING-MEAN-ONLY (separate figures)
    # No smoothing on the raw plots.
    # ----------------------------

    # Episodes: raw
    plot_line(
        x=train_df["episode_idx_global"].to_numpy(),
        y=train_df["r"].to_numpy(),
        color=ROBOCLIP_COLOR,
        xlabel="Episode index (global across training workers)",
        ylabel="RoboCLIP training reward (monitor r)",
        title="RoboCLIP training reward vs episodes (raw)",
        out_dir=out_dir,
        fname="roboclip_train_reward_vs_episodes_raw.png"
    )

    # Episodes: rolling-mean-only (separate plot)
    plot_line_rolling_mean_only(
        x=train_df["episode_idx_global"].to_numpy(),
        y=train_df["r"].to_numpy(),
        color=ROBOCLIP_COLOR,
        window=args.train_rolling_window,
        xlabel="Episode index (global across training workers)",
        ylabel="RoboCLIP training reward (rolling mean)",
        title="RoboCLIP training reward vs episodes",
        out_dir=out_dir,
        fname="roboclip_train_reward_vs_episodes_rolling_mean.png"
    )

    # Approx timesteps: raw
    plot_line(
        x=train_df["approx_timesteps"].to_numpy(),
        y=train_df["r"].to_numpy(),
        color=ROBOCLIP_COLOR,
        xlabel="Approx training timesteps (cumsum of episode lengths from training monitors)",
        ylabel="RoboCLIP training reward (monitor r)",
        title="RoboCLIP training reward vs approx timesteps (raw)",
        out_dir=out_dir,
        fname="roboclip_train_reward_vs_approx_timesteps_raw.png"
    )

    # Approx timesteps: rolling-mean-only
    plot_line_rolling_mean_only(
        x=train_df["approx_timesteps"].to_numpy(),
        y=train_df["r"].to_numpy(),
        color=ROBOCLIP_COLOR,
        window=args.train_rolling_window,
        xlabel="Approx training timesteps (cumsum of episode lengths from training monitors)",
        ylabel="RoboCLIP training reward (rolling mean)",
        title="RoboCLIP training reward vs approx timesteps",
        out_dir=out_dir,
        fname="roboclip_train_reward_vs_approx_timesteps_rolling_mean.png"
    )

    # Optional: eval monitor dense reward (sanity) - raw + rolling-mean-only, separate
    if args.also_plot_eval_monitor_dense:
        if len(eval_df) == 0:
            print("[Info] No eval monitor rows found (ranks 10..). Skipping eval monitor plots.")
        else:
            eval_df = make_global_episode_index(eval_df)
            eval_df = add_approx_timesteps(eval_df)

            plot_line(
                x=eval_df["episode_idx_global"].to_numpy(),
                y=eval_df["r"].to_numpy(),
                color=EVAL_COLOR,
                xlabel="Episode index (global across eval workers)",
                ylabel="Dense reward from eval monitors (monitor r)",
                title="Dense reward (from eval monitor.csv) vs episodes (raw, sanity check)",
                out_dir=out_dir,
                fname="dense_eval_reward_from_monitor_vs_episodes_raw.png"
            )

            plot_line_rolling_mean_only(
                x=eval_df["episode_idx_global"].to_numpy(),
                y=eval_df["r"].to_numpy(),
                color=EVAL_COLOR,
                window=args.train_rolling_window,
                xlabel="Episode index (global across eval workers)",
                ylabel="Dense reward from eval monitors (rolling mean)",
                title="Dense reward (from eval monitor.csv) vs episodes (sanity check)",
                out_dir=out_dir,
                fname="dense_eval_reward_from_monitor_vs_episodes_rolling_mean.png"
            )

    # ----------------------------
    # Combined plot (two y-axes) - NO SMOOTHING
    # ----------------------------
    if args.combined and have_eval:
        rob_x = train_df["approx_timesteps"].to_numpy()
        rob_y = train_df["r"].to_numpy()  # raw

        eval_y = results.mean(axis=1)     # raw mean (no smoothing)

        fig, ax1 = plt.subplots()
        ax1.plot(rob_x, rob_y, color=ROBOCLIP_COLOR, label="RoboCLIP training reward (raw)")
        ax1.set_xlabel("Training timesteps (RoboCLIP axis uses approx cumsum from monitors)")
        ax1.set_ylabel("RoboCLIP training reward", color=ROBOCLIP_COLOR)
        ax1.tick_params(axis="y", labelcolor=ROBOCLIP_COLOR)

        ax2 = ax1.twinx()
        ax2.plot(timesteps, eval_y, color=EVAL_COLOR, label="Dense eval mean reward (raw)")
        ax2.set_ylabel("Dense eval mean reward", color=EVAL_COLOR)
        ax2.tick_params(axis="y", labelcolor=EVAL_COLOR)

        plt.title("RoboCLIP training reward vs Dense eval reward (raw)")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

        savefig(out_dir, "combined_roboclip_train_vs_dense_eval_raw.png")

    # ----------------------------
    # Report
    # ----------------------------
    print(f"[OK] Loaded monitor rows: total={len(mdf)}, train={len(train_df)}, eval={len(eval_df)}, other={len(other_df)}")
    if len(other_df) > 0:
        ranks = sorted([r for r in other_df["rank"].dropna().unique().tolist()])
        print(f"[Info] Found monitor ranks outside expected train/eval sets: {ranks}")
    print(f"[Done] Plots saved to: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
