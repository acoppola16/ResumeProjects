#!/usr/bin/env python3
"""Crop masked images to head and shoulders.

This script scans a directory for masked PNG images and crops them to
show only the person's face and shoulders. The cropping is based on the
bounding box of the non‑transparent area (alpha channel) or the mask.
"""

import argparse
import os
import cv2
import numpy as np


def crop_head_shoulders(
    image_path: str,
    dest_path: str,
    top_fraction: float = 0.6,
    size: int | None = None,
    keep_alpha: bool = False,
) -> None:
    """Crop the top portion of the person and save to ``dest_path``.

    Parameters
    ----------
    image_path : str
        Path to the masked PNG image (with alpha channel).
    dest_path : str
        Where to write the cropped image.
    top_fraction : float
        Fraction of the person's bounding box height to keep. A value of
        ``0.6`` will keep the top 60% (head and shoulders).
    size : int, optional
        If provided, resize the cropped result to ``size`` x ``size`` pixels.
    keep_alpha : bool
        Keep the alpha channel in the output. By default it is dropped so the
        image is RGB.
    """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    if img.shape[2] == 4:
        mask = img[:, :, 3]
    else:
        mask = (img[:, :, :3].sum(axis=2) > 0).astype(np.uint8) * 255

    ys, xs = np.where(mask > 0)
    if len(ys) == 0 or len(xs) == 0:
        return

    top, bottom = ys.min(), ys.max()
    left, right = xs.min(), xs.max()

    height = bottom - top
    new_bottom = top + int(height * top_fraction)
    new_bottom = min(new_bottom, img.shape[0] - 1)

    cropped = img[top:new_bottom, left:right]
    if not keep_alpha and cropped.shape[2] == 4:
        cropped = cropped[:, :, :3]
    if size is not None:
        cropped = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    cv2.imwrite(dest_path, cropped)


def process_directory(
    src_dir: str,
    dest_dir: str,
    top_fraction: float = 0.6,
    size: int | None = None,
    keep_alpha: bool = False,
) -> None:
    """Process all images in ``src_dir`` and write cropped images to ``dest_dir``."""
    for fname in os.listdir(src_dir):
        if not fname.lower().endswith(".png"):
            continue
        src = os.path.join(src_dir, fname)
        dest = os.path.join(dest_dir, fname)
        crop_head_shoulders(src, dest, top_fraction, size=size, keep_alpha=keep_alpha)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crop dataset to head and shoulders and resize to a square"
    )
    parser.add_argument("src", help="Directory with masked PNG images")
    parser.add_argument("dest", help="Directory to write cropped images")
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.6,
        help="Fraction of body to keep (default: 0.6)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help="Resize output to this square dimension",
    )
    parser.add_argument(
        "--keep-alpha",
        action="store_true",
        help="Keep the alpha channel in the output",
    )
    args = parser.parse_args()
    process_directory(
        args.src,
        args.dest,
        top_fraction=args.fraction,
        size=args.size,
        keep_alpha=args.keep_alpha,
    )
