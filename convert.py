#!/usr/bin/env python3
"""
Convert all STL files in the current folder to a top-down 'raycast' PNG.

We project each triangle onto the XY plane and rasterise it:
if any part of the model covers a pixel from above, that pixel is blue.
"""

import pathlib
from typing import Iterable, Tuple

import numpy as np
from PIL import Image, ImageDraw
from stl import mesh


def find_stl_files(folder: pathlib.Path) -> Iterable[pathlib.Path]:
    """Yield all STL files in the folder (case-insensitive extension)."""
    patterns = ("*.stl", "*.STL")
    for pattern in patterns:
        for stl_path in folder.glob(pattern):
            if stl_path.is_file():
                yield stl_path


def load_xy_triangles(stl_path: pathlib.Path) -> np.ndarray:
    """
    Load STL and return (N, 3, 2) array of XY triangle vertices.
    Returns empty array on failure.
    """
    try:
        m = mesh.Mesh.from_file(str(stl_path))
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Failed to load {stl_path.name}: {exc}")
        return np.empty((0, 3, 2), dtype=float)

    tris3 = m.vectors  # shape (N, 3, 3)
    if tris3.size == 0:
        print(f"[WARN] {stl_path.name}: no triangles found")
        return np.empty((0, 3, 2), dtype=float)

    tris2 = tris3[:, :, :2]  # drop Z, keep XY
    tris2 = tris2.reshape(-1, 3, 2)

    # Drop any triangles with NaNs
    mask = ~np.isnan(tris2).any(axis=(1, 2))
    tris2 = tris2[mask]

    if tris2.size == 0:
        print(f"[WARN] {stl_path.name}: all triangles invalid in XY")
        return np.empty((0, 3, 2), dtype=float)

    return tris2


def compute_xy_bounds(tris2: np.ndarray) -> Tuple[float, float, float, float]:
    """Compute padded XY bounds for triangles."""
    xs = tris2[:, :, 0].ravel()
    ys = tris2[:, :, 1].ravel()

    x_min = float(xs.min())
    x_max = float(xs.max())
    y_min = float(ys.min())
    y_max = float(ys.max())

    dx = max(x_max - x_min, 1e-6)
    dy = max(y_max - y_min, 1e-6)

    pad_x = dx * 0.05
    pad_y = dy * 0.05

    return (
        x_min - pad_x,
        x_max + pad_x,
        y_min - pad_y,
        y_max + pad_y,
    )


def world_to_image_coords(
    tris2: np.ndarray,
    bounds: Tuple[float, float, float, float],
    width: int,
    height: int,
) -> np.ndarray:
    """
    Map world XY triangle vertices to image pixel coords.

    Image origin is top-left: (0, 0) is top, (width-1, height-1) is bottom-right.
    """
    x_min, x_max, y_min, y_max = bounds
    xs = tris2[:, :, 0]
    ys = tris2[:, :, 1]

    # Normalise to [0, 1]
    nx = (xs - x_min) / max(x_max - x_min, 1e-6)
    ny = (ys - y_min) / max(y_max - y_min, 1e-6)

    # Map to pixel space
    px = nx * (width - 1)
    # Flip Y so that larger world-Y is lower in the image
    py = (1.0 - ny) * (height - 1)

    pix_tris = np.stack([px, py], axis=-1)

    return pix_tris


def rasterise_triangles_to_png(
    stl_path: pathlib.Path,
    pix_tris: np.ndarray,
    width: int = 1024,
    height: int = 1024,
) -> None:
    """
    Rasterise triangles into a PNG.

    If any triangle covers a pixel from above, that pixel becomes blue.
    Background is transparent.
    """
    if pix_tris.size == 0:
        print(f"[SKIP] {stl_path.name}: no triangles to rasterise")
        return

    out_path = stl_path.with_suffix(".png")

    # RGBA image, transparent background
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    blue = (0, 0, 255, 255)

    for tri in pix_tris:
        # tri is (3, 2) of float pixel coords
        points = [(float(tri[i, 0]), float(tri[i, 1])) for i in range(3)]
        try:
            draw.polygon(points, fill=blue)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed to draw triangle in {stl_path.name}: {exc}")

    try:
        img.save(out_path, format="PNG")
        print(f"[OK]   {stl_path.name} -> {out_path.name}")
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Failed to save {out_path.name}: {exc}")


def process_stl(stl_path: pathlib.Path) -> None:
    """Full pipeline for one STL -> PNG using top-down triangle rasterisation."""
    print(f"[INFO] Processing {stl_path.name} ...")

    tris2 = load_xy_triangles(stl_path)
    if tris2.size == 0:
        return

    bounds = compute_xy_bounds(tris2)
    pix_tris = world_to_image_coords(tris2, bounds, width=1024, height=1024)
    rasterise_triangles_to_png(stl_path, pix_tris, width=1024, height=1024)


def main() -> None:
    """Process all STL files in the current directory."""
    folder = pathlib.Path(".").resolve()
    stl_files = list(find_stl_files(folder))

    if not stl_files:
        print(f"[INFO] No STL files found in {folder}")
        return

    print(f"[INFO] Found {len(stl_files)} STL file(s) in {folder}")
    for stl_path in stl_files:
        process_stl(stl_path)


if __name__ == "__main__":
    main()
