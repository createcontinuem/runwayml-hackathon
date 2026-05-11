"""
extract_masks.py — Subject extraction from organism videos

For each .mp4 in an experiment's videos/ and growth/ folders:
  1. Extracts three frames (start, middle, end), picks the sharpest
  2. Runs rembg to remove background
  3. Saves transparent PNG to experiment's masks/ folder
  4. Updates garden_state.json with mask paths

Usage:
  python extract_masks.py
  python extract_masks.py --experiment bacteria
"""

import json
import argparse
import cv2
import numpy as np
from pathlib import Path
from rembg import remove
from PIL import Image

EXPERIMENTS_DIR = Path("output/experiments")
STATE_FILE = Path("garden_state.json")


def experiment_dirs(name: str) -> dict:
    base = EXPERIMENTS_DIR / name
    dirs = {
        "videos": base / "videos",
        "growth": base / "growth",
        "masks": base / "masks",
    }
    dirs["masks"].mkdir(parents=True, exist_ok=True)
    return dirs


def sharpness(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def extract_best_frame(video_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    candidates = []
    for pos in [total // 4, total // 2, (3 * total) // 4]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, pos - 1))
        ret, frame = cap.read()
        if ret:
            candidates.append(frame)

    cap.release()
    if not candidates:
        raise RuntimeError(f"No frames extracted from {video_path}")

    return max(candidates, key=sharpness)


def remove_background(frame: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return remove(Image.fromarray(rgb))


def process_video(video_path: Path, masks_dir: Path) -> Path:
    mask_path = masks_dir / f"{video_path.stem}_mask.png"

    if mask_path.exists():
        print(f"  cached   : {mask_path.name}")
        return mask_path

    print(f"  extract  : {video_path.name}")
    frame = extract_best_frame(video_path)

    print(f"  rembg    : removing background...")
    result = remove_background(frame)
    result.save(mask_path)

    print(f"  saved    : {mask_path.name}")
    return mask_path


def main():
    parser = argparse.ArgumentParser(description="Extract subject masks from experiment videos")
    parser.add_argument("--experiment", default="default", help="Experiment name (default: 'default')")
    args = parser.parse_args()

    if not STATE_FILE.exists():
        print("garden_state.json not found — run garden.py first")
        return

    dirs = experiment_dirs(args.experiment)

    with open(STATE_FILE) as f:
        state = json.load(f)

    videos = (
        list(dirs["videos"].glob("*.mp4")) +
        list(dirs["growth"].glob("*.mp4"))
    )

    if not videos:
        print(f"No videos found in {EXPERIMENTS_DIR / args.experiment}")
        return

    mask_map: dict[str, str] = {}

    print(f"Experiment : {args.experiment}")
    print(f"Processing {len(videos)} videos...\n")

    for video_path in sorted(videos):
        try:
            mask_path = process_video(video_path, dirs["masks"])
            mask_map[video_path.stem] = str(mask_path).replace("\\", "/")
            print()
        except Exception as e:
            print(f"  error    : {e}\n")

    for org in state["organisms"]:
        stem = Path(org["video"]).stem
        if stem in mask_map:
            org["mask"] = mask_map[stem]

    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

    print(f"Updated {STATE_FILE} with {len(mask_map)} masks.")


if __name__ == "__main__":
    main()
