"""
garden.py — Digital garden state builder + local server

Scans an experiment's videos/ and growth/ folders for .mp4 files,
assigns positions, writes garden_state.json, then serves on localhost:8080.

Usage:
  python garden.py
  python garden.py --experiment bacteria
  open http://localhost:8080/garden.html
"""

import json
import math
import random
import argparse
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler

EXPERIMENTS_DIR = Path("output/experiments")
STATE_FILE = Path("garden_state.json")

random.seed(42)


def scan_organisms(experiment: str) -> list[dict]:
    base = EXPERIMENTS_DIR / experiment
    videos_dir = base / "videos"
    growth_dir = base / "growth"
    organisms = []

    for mp4 in sorted(videos_dir.glob("*.mp4")):
        organisms.append({
            "id": mp4.stem,
            "species": mp4.stem,
            "generation": 0,
            "video": str(mp4).replace("\\", "/"),
            "type": "single",
        })

    for mp4 in sorted(growth_dir.glob("*.mp4")):
        stem = mp4.stem
        species = stem.rsplit("_gen", 1)[0] if "_gen" in stem else stem
        gen = int(stem.rsplit("_gen", 1)[1]) if "_gen" in stem else 0
        organisms.append({
            "id": stem,
            "species": species,
            "generation": gen,
            "video": str(mp4).replace("\\", "/"),
            "type": "growth",
        })

    return organisms


def assign_positions(organisms: list[dict]) -> list[dict]:
    count = len(organisms)
    cols = math.ceil(math.sqrt(count))
    spacing = 3.5

    for i, org in enumerate(organisms):
        col = i % cols
        row = i // cols
        org["position"] = {
            "x": (col - cols / 2) * spacing + random.uniform(-0.6, 0.6),
            "y": random.uniform(-0.5, 0.5),
            "z": (row - cols / 2) * spacing + random.uniform(-0.6, 0.6),
        }
        org["float_speed"] = random.uniform(0.3, 0.8)
        org["float_amplitude"] = random.uniform(0.1, 0.3)
        org["phase_offset"] = random.uniform(0, math.pi * 2)

    return organisms


def build_state(experiment: str):
    organisms = scan_organisms(experiment)
    if not organisms:
        print(f"No videos found in {EXPERIMENTS_DIR / experiment}")
        return

    organisms = assign_positions(organisms)
    state = {"experiment": experiment, "organism_count": len(organisms), "organisms": organisms}

    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

    print(f"Experiment : {experiment}")
    print(f"Organisms  : {len(organisms)} written to {STATE_FILE}")
    for org in organisms:
        print(f"  [{org['type']:6}] gen{org['generation']} — {org['id']}")


class CORSHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def log_message(self, format, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description="Build garden state and serve")
    parser.add_argument("--experiment", default="default", help="Experiment name (default: 'default')")
    parser.add_argument("--build-only", action="store_true", help="Build state file and exit without serving")
    args = parser.parse_args()

    build_state(args.experiment)

    if args.build_only:
        return

    server = HTTPServer(("localhost", 8080), CORSHandler)
    print("\nServing at http://localhost:8080/garden.html\n")
    server.serve_forever()


if __name__ == "__main__":
    main()
