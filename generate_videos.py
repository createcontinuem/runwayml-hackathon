import io
import os
import sys
import base64
import json
import argparse
import urllib.request
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
from runwayml import RunwayML, TaskFailedError

load_dotenv()

IMAGES_DIR = Path("references/images/photos")
EXPERIMENTS_DIR = Path("output/experiments")
PROMPTS_FILE = IMAGES_DIR / "prompts.json"
MULTI_CONFIG_FILE = IMAGES_DIR / "multi.json"

SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".webp"}

CREDITS = {
    "video_per_second": {"gen4.5": 12, "gen4_turbo": 5, "gen3a_turbo": 5, "veo3": 40},
    "image": {"gen4_image": 5, "gen4_image_turbo": 2},
}

DEFAULT_COMPOSITE_PROMPT = "fused organic forms, continuem aesthetic, microorganism, dark background"


def experiment_dirs(name: str) -> dict:
    base = EXPERIMENTS_DIR / name
    dirs = {
        "videos": base / "videos",
        "composites": base / "composites",
        "results": base / "results.json",
    }
    for key in ["videos", "composites"]:
        dirs[key].mkdir(parents=True, exist_ok=True)
    return dirs


def estimate_credits(model: str, duration: int | None = None) -> int:
    if duration is not None:
        return CREDITS["video_per_second"].get(model, 0) * duration
    return CREDITS["image"].get(model, 0)


def load_prompts() -> dict:
    if PROMPTS_FILE.exists():
        with open(PROMPTS_FILE) as f:
            return json.load(f)
    return {}


MAX_IMAGE_SIZE = (1280, 720)


def image_to_data_uri(path: Path) -> str:
    img = Image.open(path).convert("RGB")
    img.thumbnail(MAX_IMAGE_SIZE, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def download_video(url: str, dest: Path):
    urllib.request.urlretrieve(url, dest)


def generate_single(client: RunwayML, image_path: Path, prompt: str, tracker: dict, dirs: dict) -> dict:
    credits = estimate_credits("gen4.5", duration=5)
    print(f"[{image_path.name}]")
    print(f"  prompt   : {prompt}")
    print(f"  credits  : ~{credits}")

    try:
        task = client.image_to_video.create(
            model="gen4.5",
            prompt_image=image_to_data_uri(image_path),
            prompt_text=prompt,
            ratio="1280:720",
            duration=5,
        ).wait_for_task_output()

        tracker["credits_used"] += credits
        video_url = task.output[0]
        out_file = dirs["videos"] / f"{image_path.stem}.mp4"
        download_video(video_url, out_file)

        print(f"  saved    : {out_file}")
        print(f"  total    : {tracker['credits_used']} credits used so far\n")
        return {"image": image_path.name, "prompt": prompt, "url": video_url,
                "output": str(out_file), "credits_used": credits, "status": "success"}

    except TaskFailedError as e:
        print(f"  failed : {e.task_details}\n")
        return {"image": image_path.name, "prompt": prompt, "status": "failed", "error": str(e.task_details)}
    except Exception as e:
        print(f"  error  : {e}\n")
        return {"image": image_path.name, "prompt": prompt, "status": "error", "error": str(e)}


def generate_multi(client: RunwayML, image_paths: list[Path], composite_prompt: str, motion_prompt: str, out_name: str, tracker: dict, dirs: dict) -> dict:
    print(f"[multi -> {out_name}]")
    print(f"  references   : {[p.name for p in image_paths]}")
    print(f"  composite    : {composite_prompt}")
    print(f"  motion       : {motion_prompt}")

    reference_images = []
    tagged_prompt = composite_prompt
    for i, path in enumerate(image_paths):
        tag = f"ref{i}"
        reference_images.append({"uri": image_to_data_uri(path), "tag": tag})
        if f"@{tag}" not in tagged_prompt:
            tagged_prompt = f"@{tag} {tagged_prompt}"

    image_credits = estimate_credits("gen4_image")
    print(f"  step 1/2     : generating composite image... (~{image_credits} credits)")
    try:
        composite_task = client.text_to_image.create(
            model="gen4_image",
            prompt_text=tagged_prompt,
            reference_images=reference_images,
            ratio="1280:720",
        ).wait_for_task_output()
    except Exception as e:
        print(f"  failed at step 1: {e}\n")
        return {"name": out_name, "status": "failed", "error": str(e)}

    tracker["credits_used"] += image_credits
    composite_url = composite_task.output[0]
    composite_file = dirs["composites"] / f"{out_name}_composite.jpg"
    urllib.request.urlretrieve(composite_url, composite_file)
    print(f"  composite    : {composite_file} ({image_credits} credits)")

    video_credits = estimate_credits("gen4.5", duration=5)
    print(f"  step 2/2     : animating composite... (~{video_credits} credits)")
    try:
        video_task = client.image_to_video.create(
            model="gen4.5",
            prompt_image=composite_url,
            prompt_text=motion_prompt,
            ratio="1280:720",
            duration=5,
        ).wait_for_task_output()
    except Exception as e:
        print(f"  failed at step 2: {e}\n")
        return {"name": out_name, "status": "failed", "error": str(e)}

    tracker["credits_used"] += video_credits
    video_url = video_task.output[0]
    out_file = dirs["videos"] / f"{out_name}.mp4"
    urllib.request.urlretrieve(video_url, out_file)
    step_total = image_credits + video_credits
    print(f"  saved        : {out_file}")
    print(f"  credits      : {step_total} this combo | {tracker['credits_used']} total\n")

    return {"mode": "multi", "references": [p.name for p in image_paths],
            "composite_prompt": tagged_prompt, "motion_prompt": motion_prompt,
            "composite": str(composite_file), "url": video_url,
            "output": str(out_file), "credits_used": step_total, "status": "success"}


def main():
    if not os.getenv("RUNWAYML_API_SECRET"):
        raise EnvironmentError("RUNWAYML_API_SECRET not set — check your .env file")

    parser = argparse.ArgumentParser(description="Generate videos from reference images via Runway API")
    parser.add_argument("--experiment", default="default", help="Experiment name (default: 'default')")
    parser.add_argument("--images", help="Comma-separated image filenames from photos dir (default: all)")
    parser.add_argument("--prompt", help="Motion prompt override (overrides prompts.json)")
    parser.add_argument("--composite-prompt", help="Composite prompt for multi mode (overrides multi.json)")
    parser.add_argument("--multi-only", action="store_true", help="Skip single-image generation")
    parser.add_argument("--single-only", action="store_true", help="Skip multi-image combos")
    args = parser.parse_args()

    dirs = experiment_dirs(args.experiment)
    print(f"Experiment : {args.experiment}")
    print(f"Output     : {EXPERIMENTS_DIR / args.experiment}\n")

    client = RunwayML()
    saved_prompts = load_prompts()
    default_prompt = args.prompt or saved_prompts.get("_default", "slow organic pulse, floating, breathing, subtle motion")
    tracker = {"credits_used": 0}
    results = []

    if args.images:
        images = [IMAGES_DIR / f.strip() for f in args.images.split(",")]
        missing = [str(p) for p in images if not p.exists()]
        if missing:
            print(f"Images not found: {missing}")
            sys.exit(1)
    else:
        images = sorted(p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in SUPPORTED_FORMATS)

    if not args.multi_only:
        single_credits = estimate_credits("gen4.5", duration=5)
        print(f"Found {len(images)} image(s) — ~{single_credits} credits each\n")
        for image_path in images:
            prompt = args.prompt or saved_prompts.get(image_path.name, default_prompt)
            results.append(generate_single(client, image_path, prompt, tracker, dirs))

    if not args.single_only:
        if args.images and args.prompt:
            out_name = "multi_" + "_".join(p.stem for p in images)
            composite = args.composite_prompt or DEFAULT_COMPOSITE_PROMPT
            results.append(generate_multi(client, images, composite, args.prompt, out_name, tracker, dirs))
        elif MULTI_CONFIG_FILE.exists() and not args.images:
            with open(MULTI_CONFIG_FILE) as f:
                combos = json.load(f)
            for combo in combos:
                name = combo.get("name", "multi_output")
                image_files = [IMAGES_DIR / fname for fname in combo["images"]]
                missing = [str(p) for p in image_files if not p.exists()]
                if missing:
                    print(f"[{name}] skipped — missing: {missing}\n")
                    results.append({"name": name, "status": "skipped", "missing": missing})
                    continue
                composite = args.composite_prompt or combo.get("composite_prompt", DEFAULT_COMPOSITE_PROMPT)
                motion = args.prompt or combo.get("motion_prompt", default_prompt)
                results.append(generate_multi(client, image_files, composite, motion, name, tracker, dirs))

    with open(dirs["results"], "w") as f:
        json.dump(results, f, indent=2)

    success = sum(1 for r in results if r.get("status") == "success")
    print(f"Done — {success}/{len(results)} succeeded.")
    print(f"Total credits used : {tracker['credits_used']}")
    print(f"Results            : {dirs['results']}")


if __name__ == "__main__":
    main()
