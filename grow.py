"""
grow.py — Iterative biological growth pipeline

Takes a seed video, extracts its last frame, and feeds it into Runway's
image_to_video repeatedly — each generation evolving from the previous output.

Usage:
  python grow.py --input output/experiments/default/videos/soul_fusion_239_247.mp4
  python grow.py --input ... --experiment bacteria --iterations 4 --prompt "..."
"""

import io
import os
import sys
import base64
import argparse
import json
import urllib.request
from pathlib import Path
from datetime import datetime

import cv2
from PIL import Image
from dotenv import load_dotenv
from runwayml import RunwayML, TaskFailedError

load_dotenv()

EXPERIMENTS_DIR = Path("output/experiments")
CREDITS_PER_VIDEO = 60  # gen4.5 @ 12 credits/sec × 5sec

DEFAULT_PROMPT = (
    "biological cell division and growth, bacteria multiplying under microscope, "
    "flagella propulsion, colony expanding, dark field microscopy, "
    "bioluminescent green fluorescence, membrane dividing and replicating"
)

MAX_IMAGE_SIZE = (1280, 720)


def experiment_dirs(name: str) -> dict:
    base = EXPERIMENTS_DIR / name
    dirs = {
        "growth": base / "growth",
        "log": base / "growth_log.json",
    }
    dirs["growth"].mkdir(parents=True, exist_ok=True)
    return dirs


def extract_last_frame(video_path: Path, output_dir: Path) -> Path:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("Failed to extract last frame")

    frame_path = output_dir / f"{video_path.stem}_last_frame.jpg"
    cv2.imwrite(str(frame_path), frame)
    return frame_path


def image_to_data_uri(path: Path) -> str:
    img = Image.open(path).convert("RGB")
    img.thumbnail(MAX_IMAGE_SIZE, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


CREDITS_PER_IMAGE = 5  # gen4_image @ 720p


def generate_generation(client: RunwayML, frame_path: Path, prompt: str, gen_num: int, seed_name: str, output_dir: Path, reference: Path | None = None) -> dict:
    out_name = f"{seed_name}_gen{gen_num:02d}"

    print(f"  [Generation {gen_num}]")
    print(f"  frame    : {frame_path.name}")
    if reference:
        print(f"  reference: {reference.name}")
    print(f"  prompt   : {prompt}")

    prompt_image_uri = image_to_data_uri(frame_path)

    # Two-step chain when reference is provided
    if reference:
        composite_credits = CREDITS_PER_IMAGE
        print(f"  step 1/2 : blending with reference... (~{composite_credits} credits)")

        ref_images = [
            {"uri": image_to_data_uri(frame_path), "tag": "ref0"},
            {"uri": image_to_data_uri(reference), "tag": "ref1"},
        ]
        composite_prompt = f"@ref0 @ref1 {prompt}"
        composite_task = client.text_to_image.create(
            model="gen4_image",
            prompt_text=composite_prompt,
            reference_images=ref_images,
            ratio="1280:720",
        ).wait_for_task_output()

        composite_url = composite_task.output[0]
        composite_file = output_dir / f"{out_name}_composite.jpg"
        urllib.request.urlretrieve(composite_url, composite_file)
        print(f"  composite: {composite_file.name} ({composite_credits} credits)")
        prompt_image_uri = composite_url
        video_credits = CREDITS_PER_VIDEO
        total_credits = composite_credits + video_credits
        print(f"  step 2/2 : animating... (~{video_credits} credits)")
    else:
        total_credits = CREDITS_PER_VIDEO
        print(f"  credits  : ~{total_credits}")

    task = client.image_to_video.create(
        model="gen4.5",
        prompt_image=prompt_image_uri,
        prompt_text=prompt,
        ratio="1280:720",
        duration=5,
    ).wait_for_task_output()

    video_url = task.output[0]
    out_file = output_dir / f"{out_name}.mp4"
    urllib.request.urlretrieve(video_url, out_file)

    print(f"  saved    : {out_file}")
    print(f"  credits  : {total_credits} this gen\n")
    return {
        "generation": gen_num,
        "name": out_name,
        "source_frame": str(frame_path),
        "reference": str(reference) if reference else None,
        "prompt": prompt,
        "url": video_url,
        "output": str(out_file),
        "credits_used": total_credits,
        "status": "success",
    }


def main():
    if not os.getenv("RUNWAYML_API_SECRET"):
        raise EnvironmentError("RUNWAYML_API_SECRET not set — check your .env file")

    parser = argparse.ArgumentParser(description="Iterative biological growth pipeline")
    parser.add_argument("--input", required=True, help="Seed video path")
    parser.add_argument("--experiment", default="default", help="Experiment name (default: 'default')")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Growth prompt (applied each generation)")
    parser.add_argument("--iterations", type=int, default=3, help="Number of growth generations (default: 3)")
    parser.add_argument("--reference", help="Reference image path to blend into each generation")
    args = parser.parse_args()

    seed_path = Path(args.input)
    if not seed_path.exists():
        print(f"Input video not found: {seed_path}")
        sys.exit(1)

    reference = None
    if args.reference:
        reference = Path(args.reference)
        if not reference.exists():
            print(f"Reference image not found: {reference}")
            sys.exit(1)

    dirs = experiment_dirs(args.experiment)
    client = RunwayML()
    seed_name = seed_path.stem
    total_credits = 0
    log = {
        "experiment": args.experiment,
        "seed": str(seed_path),
        "prompt": args.prompt,
        "iterations": args.iterations,
        "started": datetime.now().isoformat(),
        "generations": [],
    }

    credits_per_gen = CREDITS_PER_VIDEO + (CREDITS_PER_IMAGE if reference else 0)
    print(f"\nExperiment      : {args.experiment}")
    print(f"Growth pipeline : {seed_path.name}")
    print(f"Reference       : {reference.name if reference else 'none'}")
    print(f"Iterations      : {args.iterations}")
    print(f"Est. credits    : ~{args.iterations * credits_per_gen}")
    print(f"Output          : {dirs['growth']}\n")

    current_video = seed_path

    for i in range(1, args.iterations + 1):
        try:
            frame_path = extract_last_frame(current_video, dirs["growth"])
            result = generate_generation(client, frame_path, args.prompt, i, seed_name, dirs["growth"], reference)
            log["generations"].append(result)
            total_credits += result["credits_used"]
            current_video = Path(result["output"])

        except TaskFailedError as e:
            print(f"  failed : {e.task_details}\n")
            log["generations"].append({"generation": i, "status": "failed", "error": str(e.task_details)})
            break
        except Exception as e:
            print(f"  error  : {e}\n")
            log["generations"].append({"generation": i, "status": "error", "error": str(e)})
            break

    log["total_credits_used"] = total_credits
    with open(dirs["log"], "w") as f:
        json.dump(log, f, indent=2)

    success = sum(1 for g in log["generations"] if g.get("status") == "success")
    print(f"Done — {success}/{args.iterations} generations succeeded.")
    print(f"Credits used : {total_credits}")
    print(f"Output       : {dirs['growth']}/")
    print(f"Log          : {dirs['log']}")


if __name__ == "__main__":
    main()
