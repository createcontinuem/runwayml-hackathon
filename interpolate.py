"""
interpolate.py — Frame interpolation between two images via Runway API (Option 3)

Step 1: text_to_image blends both images into a visual midpoint (~5 credits)
Step 2: image_to_video animates the midpoint as a dissolve (~60 credits)
Total: ~65 credits per pair

Usage:
  # Single pair
  python interpolate.py --a img_a.jpg --b img_b.jpg

  # Sequence — chains A→B, B→C, C→D, ... and concatenates into one video
  python interpolate.py --sequence "a.jpg,b.jpg,c.jpg,d.jpg"
  python interpolate.py --sequence "a.jpg,b.jpg,c.jpg" --stitch
"""

import io
import os
import subprocess
import base64
import argparse
import urllib.request
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
from runwayml import RunwayML, TaskFailedError

load_dotenv()

EXPERIMENTS_DIR = Path("output/experiments")
MAX_IMAGE_SIZE = (1280, 720)

BLEND_PROMPT  = "@ref0 transitioning into @ref1, visual midpoint between two images"
MOTION_PROMPT = "smooth crossfade transition, one image dissolving into another"


def image_to_data_uri(path: Path) -> str:
    img = Image.open(path).convert("RGB")
    img.thumbnail(MAX_IMAGE_SIZE, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def download(url: str, dest: Path):
    urllib.request.urlretrieve(url, dest)
    print(f"  saved : {dest}")


def interpolate(client: RunwayML, image_a: Path, image_b: Path, out_dir: Path, duration: int) -> Path | None:
    """Interpolate between two images. Returns the output video path, or None on failure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{image_a.stem}__to__{image_b.stem}"

    # ── Step 1: blend ────────────────────────────────────────────────────────
    print(f"\nStep 1/2  blend  (~5 credits)")
    print(f"  ref0 : {image_a.name}")
    print(f"  ref1 : {image_b.name}")

    try:
        composite_task = client.text_to_image.create(
            model="gen4_image",
            prompt_text=BLEND_PROMPT,
            reference_images=[
                {"uri": image_to_data_uri(image_a), "tag": "ref0"},
                {"uri": image_to_data_uri(image_b), "tag": "ref1"},
            ],
            ratio="1280:720",
        ).wait_for_task_output()
    except TaskFailedError as e:
        print(f"  failed : {e.task_details}")
        return None
    except Exception as e:
        print(f"  error  : {e}")
        return None

    composite_url = composite_task.output[0]
    composite_file = out_dir / f"{stem}_blend.jpg"
    download(composite_url, composite_file)

    # ── Step 2: animate ──────────────────────────────────────────────────────
    video_credits = duration * 12
    print(f"\nStep 2/2  animate  (~{video_credits} credits)")

    try:
        video_task = client.image_to_video.create(
            model="gen4.5",
            prompt_image=composite_url,
            prompt_text=MOTION_PROMPT,
            ratio="1280:720",
            duration=duration,
        ).wait_for_task_output()
    except TaskFailedError as e:
        print(f"  failed : {e.task_details}")
        return None
    except Exception as e:
        print(f"  error  : {e}")
        return None

    video_file = out_dir / f"{stem}.mp4"
    download(video_task.output[0], video_file)
    print(f"  credits : {5 + video_credits} this pair")
    return video_file


def stitch(video_files: list[Path], out_file: Path):
    """Concatenate video files into one using ffmpeg."""
    list_file = out_file.parent / "concat_list.txt"
    list_file.write_text("\n".join(f"file '{v.resolve()}'" for v in video_files))
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(list_file), "-c", "copy", str(out_file),
    ], check=True)
    list_file.unlink()
    print(f"\nStitched : {out_file}")


def main():
    if not os.getenv("RUNWAYML_API_SECRET"):
        raise EnvironmentError("RUNWAYML_API_SECRET not set — check your .env file")

    parser = argparse.ArgumentParser(description="Frame interpolation via Runway API")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--a",         help="First image (use with --b)")
    group.add_argument("--sequence",  help="Comma-separated list of images to chain A→B→C→…")
    parser.add_argument("--b",          help="Second image (use with --a)")
    parser.add_argument("--experiment", default="interpolation", help="Output experiment name")
    parser.add_argument("--duration",   type=int, default=5, choices=[2, 4, 5, 8, 10])
    parser.add_argument("--stitch",     action="store_true", help="Concatenate sequence into one video (requires ffmpeg)")
    args = parser.parse_args()

    out_dir = EXPERIMENTS_DIR / args.experiment / "videos"
    client  = RunwayML()

    if args.sequence:
        images = [Path(p.strip()) for p in args.sequence.split(",")]
        for p in images:
            if not p.exists():
                raise FileNotFoundError(f"Image not found: {p}")

        pairs = list(zip(images, images[1:]))
        cost  = len(pairs) * (5 + args.duration * 12)
        print(f"Experiment : {args.experiment}")
        print(f"Sequence   : {' → '.join(p.name for p in images)}")
        print(f"Pairs      : {len(pairs)}")
        print(f"Est. cost  : ~{cost} credits")

        videos = []
        for i, (a, b) in enumerate(pairs):
            print(f"\n── Pair {i + 1}/{len(pairs)} ──────────────────────────────")
            result = interpolate(client, a, b, out_dir, args.duration)
            if result:
                videos.append(result)

        print(f"\nCompleted {len(videos)}/{len(pairs)} pairs")

        if args.stitch and videos:
            stitch(videos, out_dir / f"{args.experiment}_sequence.mp4")

    else:
        if not args.b:
            parser.error("--b is required when using --a")
        image_a, image_b = Path(args.a), Path(args.b)
        for p in (image_a, image_b):
            if not p.exists():
                raise FileNotFoundError(f"Image not found: {p}")

        print(f"Experiment : {args.experiment}")
        print(f"A → B      : {image_a.name} → {image_b.name}")
        print(f"Est. cost  : ~{5 + args.duration * 12} credits")
        interpolate(client, image_a, image_b, out_dir, args.duration)


if __name__ == "__main__":
    main()
