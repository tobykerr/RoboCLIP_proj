import os
import pickle
import numpy as np

INPUT_PATH = "results/subtasks_max25/grasp/init_states_tight.pkl"
OUTPUT_PATH = "results/subtasks_max25/grasp/init_states_tight_stats.txt"

def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(INPUT_PATH, "rb") as f:
        payload = pickle.load(f)

    states = payload["states"]

    lines = []
    lines.append(f"Total states saved: {len(states)}")

    if len(states) == 0:
        lines.append("No states found.")
        write_output(lines)
        return

    # Extract qpos depending on backend
    qpos_list = []
    timesteps = []

    for s in states:
        mj_state = s["mj_state"]

        # mujoco_py backend
        if hasattr(mj_state, "qpos"):
            qpos_list.append(mj_state.qpos.copy())
        else:
            # new mujoco backend
            qpos_list.append(mj_state["qpos"])

        if "t" in s:
            timesteps.append(s["t"])

    qpos = np.stack(qpos_list)

    lines.append(f"qpos shape: {qpos.shape}")

    variances = np.var(qpos, axis=0)
    lines.append(f"Mean variance across dims: {np.mean(variances)}")
    lines.append(f"Max variance across dims: {np.max(variances)}")
    lines.append(f"Min variance across dims: {np.min(variances)}")

    unique_rows = np.unique(np.round(qpos, 4), axis=0)
    lines.append(f"Approx unique states (rounded to 4dp): {len(unique_rows)}")

    if len(timesteps) > 0:
        lines.append(f"Unique saved timesteps: {sorted(set(timesteps))}")

    write_output(lines)


def write_output(lines):
    with open(OUTPUT_PATH, "w") as f:
        for line in lines:
            f.write(line + "\n")

    print(f"Stats written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()