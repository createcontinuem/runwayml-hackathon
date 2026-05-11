"""
interpolate_local.py — Compare 4 frame interpolation methods

Methods:
  crossfade   — alpha blend baseline
  flow        — bidirectional Farneback optical flow
  homography  — ORB feature matching + perspective warp interpolation
  mesh        — Delaunay triangle mesh morph (classic face-morph algorithm)

Output: one video per method in --out directory

Usage:
  python interpolate_local.py --a img_a.jpg --b img_b.jpg
  python interpolate_local.py --a img_a.jpg --b img_b.jpg --frames 60 --fps 30
  python interpolate_local.py --a img_a.jpg --b img_b.jpg --methods flow,mesh
"""

import argparse
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


# ── Image I/O ────────────────────────────────────────────────────────────────

def load(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read: {path}")
    if size:
        img = cv2.resize(img, size, interpolation=cv2.INTER_LANCZOS4)
    return img


def common_size(paths: list[Path]) -> tuple[int, int]:
    dims = [cv2.imread(str(p)).shape[:2] for p in paths]
    h = min(d[0] for d in dims)
    w = min(d[1] for d in dims)
    # Ensure even dimensions (required by H.264 encoder)
    return (w - w % 2, h - h % 2)


def save_video(frames: list[np.ndarray], out_path: Path, fps: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    tmp_dir = out_path.parent / "_tmp"
    tmp_dir.mkdir(exist_ok=True)

    for i, frame in enumerate(frames):
        cv2.imwrite(str(tmp_dir / f"{i:06d}.png"), frame)

    try:
        subprocess.run([
            "ffmpeg", "-y", "-framerate", str(fps),
            "-i", str(tmp_dir / "%06d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(out_path),
        ], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        avi = out_path.with_suffix(".avi")
        writer = cv2.VideoWriter(str(avi), cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h))
        for f in frames:
            writer.write(f)
        writer.release()
        out_path = avi
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"  → {out_path}")
    return out_path


# ── Method 1: Crossfade ───────────────────────────────────────────────────────

def method_crossfade(img_a: np.ndarray, img_b: np.ndarray, n: int) -> list[np.ndarray]:
    frames = []
    for i in range(n):
        t = i / (n - 1)
        frames.append(cv2.addWeighted(img_a, 1 - t, img_b, t, 0))
    return frames


# ── Method 2: Optical flow ───────────────────────────────────────────────────

def _farneback(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    gs = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gd = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    return cv2.calcOpticalFlowFarneback(
        gs, gd, None,
        pyr_scale=0.5, levels=5, winsize=21,
        iterations=5, poly_n=7, poly_sigma=1.5, flags=0,
    )


def _warp_flow(img: np.ndarray, flow: np.ndarray, t: float) -> np.ndarray:
    h, w = img.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32),
                          np.arange(h, dtype=np.float32))
    return cv2.remap(img,
                     gx + t * flow[..., 0],
                     gy + t * flow[..., 1],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def method_flow(img_a: np.ndarray, img_b: np.ndarray, n: int) -> list[np.ndarray]:
    fwd = _farneback(img_a, img_b)
    bwd = _farneback(img_b, img_a)
    frames = []
    for i in range(n):
        t = i / (n - 1)
        wa = _warp_flow(img_a, fwd, t)
        wb = _warp_flow(img_b, bwd, 1 - t)
        frames.append(cv2.addWeighted(wa, 1 - t, wb, t, 0))
    return frames


# ── Method 3: Homography ─────────────────────────────────────────────────────

def method_homography(img_a: np.ndarray, img_b: np.ndarray, n: int) -> list[np.ndarray]:
    ga = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(3000)
    kp_a, des_a = orb.detectAndCompute(ga, None)
    kp_b, des_b = orb.detectAndCompute(gb, None)

    if des_a is None or des_b is None or len(kp_a) < 4 or len(kp_b) < 4:
        print("  [homography] not enough features — falling back to crossfade")
        return method_crossfade(img_a, img_b, n)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des_a, des_b), key=lambda m: m.distance)[:300]

    if len(matches) < 4:
        print("  [homography] too few matches — falling back to crossfade")
        return method_crossfade(img_a, img_b, n)

    pts_a = np.float32([kp_a[m.queryIdx].pt for m in matches])
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in matches])
    H, _ = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, 5.0)

    if H is None:
        print("  [homography] homography failed — falling back to crossfade")
        return method_crossfade(img_a, img_b, n)

    H_inv = np.linalg.inv(H)
    I = np.eye(3)
    h, w = img_a.shape[:2]

    frames = []
    for i in range(n):
        t = i / (n - 1)
        # Interpolate each matrix linearly between identity and full transform
        H_fwd = I + t * (H - I)
        H_bwd = I + (1 - t) * (H_inv - I)
        wa = cv2.warpPerspective(img_a, H_fwd, (w, h))
        wb = cv2.warpPerspective(img_b, H_bwd, (w, h))
        frames.append(cv2.addWeighted(wa, 1 - t, wb, t, 0))
    return frames


# ── Method 4: Mesh morph ─────────────────────────────────────────────────────

def _add_boundary(pts: list, w: int, h: int) -> list:
    """Add corners and edge midpoints so triangulation covers the full image."""
    boundary = [
        (0, 0), (w // 2, 0), (w - 1, 0),
        (0, h // 2), (w - 1, h // 2),
        (0, h - 1), (w // 2, h - 1), (w - 1, h - 1),
    ]
    existing = {(round(p[0]), round(p[1])) for p in pts}
    for p in boundary:
        if p not in existing:
            pts.append(p)
    return pts


def _triangulate(pts: list, w: int, h: int) -> list[tuple]:
    """Delaunay triangulation. Returns list of (i,j,k) index triples."""
    subdiv = cv2.Subdiv2D((0, 0, w, h))
    pt_map = {}
    for i, p in enumerate(pts):
        key = (round(float(p[0])), round(float(p[1])))
        pt_map[key] = i
        subdiv.insert((float(p[0]), float(p[1])))

    triangles = []
    for tri in subdiv.getTriangleList():
        verts = [
            (round(tri[0]), round(tri[1])),
            (round(tri[2]), round(tri[3])),
            (round(tri[4]), round(tri[5])),
        ]
        if any(not (0 <= v[0] < w and 0 <= v[1] < h) for v in verts):
            continue
        idxs = [pt_map.get(v, -1) for v in verts]
        if -1 not in idxs:
            triangles.append(tuple(idxs))
    return triangles


def _warp_triangle(src: np.ndarray, tri_src: np.ndarray,
                   dst: np.ndarray, tri_dst: np.ndarray):
    """Affine-warp a triangle from src into dst (in-place)."""
    r_src = cv2.boundingRect(tri_src)
    r_dst = cv2.boundingRect(tri_dst)

    tri_src_local = tri_src - np.array([r_src[0], r_src[1]], dtype=np.float32)
    tri_dst_local = tri_dst - np.array([r_dst[0], r_dst[1]], dtype=np.float32)

    crop = src[r_src[1]:r_src[1] + r_src[3], r_src[0]:r_src[0] + r_src[2]]
    if crop.size == 0:
        return

    M = cv2.getAffineTransform(tri_src_local[:3], tri_dst_local[:3])
    warped = cv2.warpAffine(crop, M, (r_dst[2], r_dst[3]))

    mask = np.zeros((r_dst[3], r_dst[2]), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(tri_dst_local), 255)

    roi = dst[r_dst[1]:r_dst[1] + r_dst[3], r_dst[0]:r_dst[0] + r_dst[2]]
    if roi.shape[:2] != warped.shape[:2]:
        return
    roi[mask > 0] = warped[mask > 0]


def method_mesh(img_a: np.ndarray, img_b: np.ndarray, n: int) -> list[np.ndarray]:
    h, w = img_a.shape[:2]
    ga = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(500)
    kp_a, des_a = orb.detectAndCompute(ga, None)
    kp_b, des_b = orb.detectAndCompute(gb, None)

    if des_a is not None and des_b is not None and len(kp_a) >= 4 and len(kp_b) >= 4:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = sorted(bf.match(des_a, des_b), key=lambda m: m.distance)[:150]
        pts_a = [[kp_a[m.queryIdx].pt[0], kp_a[m.queryIdx].pt[1]] for m in matches]
        pts_b = [[kp_b[m.trainIdx].pt[0], kp_b[m.trainIdx].pt[1]] for m in matches]
    else:
        pts_a, pts_b = [], []

    # Add boundary so triangulation covers the whole image
    for pts in (pts_a, pts_b):
        boundary = [(0,0),(w//2,0),(w-1,0),(0,h//2),(w-1,h//2),(0,h-1),(w//2,h-1),(w-1,h-1)]
        for p in boundary:
            pts.append(list(p))

    pts_a_np = np.float32(pts_a)
    pts_b_np = np.float32(pts_b)

    # Triangulate on average positions for a stable mesh across all t
    pts_mid = ((pts_a_np + pts_b_np) / 2).tolist()
    triangles = _triangulate(pts_mid, w, h)

    frames = []
    for i in range(n):
        t = i / (n - 1)
        pts_t = (1 - t) * pts_a_np + t * pts_b_np

        out_a = np.zeros_like(img_a)
        out_b = np.zeros_like(img_b)

        for tri_idx in triangles:
            i0, i1, i2 = tri_idx

            tri_a = pts_a_np[[i0, i1, i2]]
            tri_b = pts_b_np[[i0, i1, i2]]
            tri_t = pts_t[[i0, i1, i2]]

            _warp_triangle(img_a, tri_a, out_a, tri_t)
            _warp_triangle(img_b, tri_b, out_b, tri_t)

        frames.append(cv2.addWeighted(out_a, 1 - t, out_b, t, 0))

    return frames


# ── Main ─────────────────────────────────────────────────────────────────────

METHODS = {
    "crossfade":  method_crossfade,
    "flow":       method_flow,
    "homography": method_homography,
    "mesh":       method_mesh,
}


def main():
    parser = argparse.ArgumentParser(description="Compare frame interpolation methods")
    parser.add_argument("--a",       required=True, help="First image")
    parser.add_argument("--b",       required=True, help="Second image")
    parser.add_argument("--frames",  type=int, default=60, help="Frames per video (default: 60)")
    parser.add_argument("--fps",     type=int, default=30, help="Output FPS (default: 30)")
    parser.add_argument("--methods", default="crossfade,flow,homography,mesh",
                        help="Comma-separated methods to run (default: all)")
    parser.add_argument("--out",     default="output/interpolated",
                        help="Output directory (default: output/interpolated)")
    args = parser.parse_args()

    img_a_path = Path(args.a)
    img_b_path = Path(args.b)
    for p in (img_a_path, img_b_path):
        if not p.exists():
            raise FileNotFoundError(f"Not found: {p}")

    size = common_size([img_a_path, img_b_path])
    img_a = load(img_a_path, size)
    img_b = load(img_b_path, size)
    out_dir = Path(args.out)
    stem = f"{img_a_path.stem}__to__{img_b_path.stem}"

    selected = [m.strip() for m in args.methods.split(",")]
    unknown = [m for m in selected if m not in METHODS]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Choose from: {list(METHODS)}")

    print(f"A → B    : {img_a_path.name} → {img_b_path.name}")
    print(f"Size     : {size[0]}×{size[1]}")
    print(f"Frames   : {args.frames} @ {args.fps} fps  ({args.frames / args.fps:.1f}s)")
    print(f"Methods  : {selected}\n")

    for name in selected:
        print(f"[{name}]")
        frames = METHODS[name](img_a, img_b, args.frames)
        save_video(frames, out_dir / f"{stem}__{name}.mp4", args.fps)

    print(f"\nDone — {len(selected)} video(s) in {out_dir}/")


if __name__ == "__main__":
    main()
