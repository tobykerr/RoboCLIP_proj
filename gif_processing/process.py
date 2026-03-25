import os
import cv2
import numpy as np
from PIL import Image

TARGET_FRAMES = 32
CROP_SIZE = 250

def uniform_sample_indices(n, k):
    if n >= k:
        return np.linspace(0, n - 1, k).round().astype(int)
    # pad by repeating last frame
    idx = list(range(n))
    idx += [n - 1] * (k - n)
    return np.array(idx, dtype=int)

def resize_min_side(frame, min_side):
    h, w = frame.shape[:2]
    scale = min_side / min(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

def center_crop(frame, size):
    h, w = frame.shape[:2]
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    return frame[y0:y0+size, x0:x0+size]

def preprocess_video_to_gif(in_path, out_path):
    cap = cv2.VideoCapture(in_path)
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(rgb)
    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames in {in_path}")

    idx = uniform_sample_indices(len(frames), TARGET_FRAMES)

    processed = []
    for i in idx:
        f = frames[i]
        f = resize_min_side(f, CROP_SIZE)
        f = center_crop(f, CROP_SIZE)
        processed.append(Image.fromarray(f))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    processed[0].save(
        out_path,
        save_all=True,
        append_images=processed[1:],
        duration=50,  # ms per frame
        loop=0,
    )

def batch_preprocess(input_dir, output_dir):
    videos = sorted(
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".mp4", ".mov", ".m4v", ".avi"))
    )

    for i, fname in enumerate(videos, start=1):
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, f"{i}.gif")
        print(f"[{i}] {in_path} -> {out_path}")
        preprocess_video_to_gif(in_path, out_path)

if __name__ == "__main__":
    batch_preprocess(
        input_dir="./drawer-open-demos",
        output_dir="./processed_gifs",
    )
