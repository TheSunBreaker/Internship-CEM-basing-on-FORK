#!/usr/bin/env python3
"""
segmentation_metrics.py

Reusable utilities to compute Dice, Hausdorff distance (HD), and HD95 between
two folders of NIfTI binary masks (matching by filename).

Dependencies:
    - nibabel
    - numpy
    - torch
    - monai
"""

import os
import csv
import numpy as np
import nibabel as nib
import torch
from typing import List, Tuple, Dict, Optional
from monai.metrics import DiceMetric, HausdorffDistanceMetric


def list_pairs(dir_a: str, dir_b: str, exts: Tuple[str, ...] = (".nii.gz", ".nii")) -> List[Tuple[str, str, str]]:
    """
    Return a list of (filename, path_a, path_b) present in BOTH folders (matched by identical filename).
    """
    A = {f: os.path.join(dir_a, f) for f in os.listdir(dir_a) if f.endswith(exts)}
    B = {f: os.path.join(dir_b, f) for f in os.listdir(dir_b) if f.endswith(exts)}
    common = sorted(set(A.keys()) & set(B.keys()))

    if not common:
        raise ValueError("No matching filenames between the two folders.")

    missing_in_b = sorted(set(A.keys()) - set(B.keys()))
    missing_in_a = sorted(set(B.keys()) - set(A.keys()))
    if missing_in_b:
        print(f"[!] {len(missing_in_b)} file(s) only in A (ignored). e.g. {missing_in_b[:3]}")
    if missing_in_a:
        print(f"[!] {len(missing_in_a)} file(s) only in B (ignored). e.g. {missing_in_a[:3]}")

    return [(fn, A[fn], B[fn]) for fn in common]


def load_mask_as_tensor(path: str, binarize_thr: float = 0.0) -> Tuple[torch.Tensor, Tuple[float, float, float]]:
    """
    Load a NIfTI mask, binarize (> binarize_thr), and return:
        - tensor of shape (1, 1, Z, Y, X) with float32 values {0.,1.}
        - spacing as (Z, Y, X) in mm (native Python floats)
    """
    img = nib.load(path)
    arr = img.get_fdata()
    mask = (arr > binarize_thr).astype(np.float32)
    t = torch.from_numpy(mask)[None, None, ...]  # (B=1, C=1, Z, Y, X)
    
    # Convert to tuple of Python floats
    spacing_zyx = tuple(float(s) for s in img.header.get_zooms()[:3][::-1])
    
    return t, spacing_zyx



def compute_metrics_for_pair(
    gt_path: str,
    pred_path: str,
    include_background: bool = False,
    hd_percentile: int = 95
) -> Tuple[float, float, float]:
    gt_t, spacing_gt = load_mask_as_tensor(gt_path)
    pr_t, spacing_pr = load_mask_as_tensor(pred_path)

    if gt_t.shape != pr_t.shape:
        raise ValueError(f"Shape mismatch: GT{tuple(gt_t.shape)} vs Pred{tuple(pr_t.shape)}")

    if not np.allclose(spacing_gt, spacing_pr, atol=1e-5):
        print(f"[!] Spacing differs for {os.path.basename(gt_path)}: "
              f"A{spacing_gt} vs B{spacing_pr}. Using GT spacing for HD/HD{hd_percentile}.")

    # --- Dice via DiceMetric (no buffer needed; single call) ---
    dice_metric = DiceMetric(include_background=include_background, reduction="mean")
    dice = dice_metric(pr_t, gt_t).item()
    dice_metric.reset()

    # --- HD (100%) and HDp (e.g., 95) ---
    hd_metric = HausdorffDistanceMetric(include_background=include_background, percentile=100)
    hd_val = hd_metric(pr_t, gt_t, spacing=spacing_gt).item()

    hd_p_metric = HausdorffDistanceMetric(include_background=include_background, percentile=hd_percentile)
    hd_p_val = hd_p_metric(pr_t, gt_t, spacing=spacing_gt).item()

    return float(dice), float(hd_val), float(hd_p_val)



def mean_std(values: List[float]) -> Tuple[float, float]:
    """
    Return (mean, std) with ddof=1 if at least 2 values, else std=0.0.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def compute_metrics_batch(
    folder_a: str,
    folder_b: str,
    include_background: bool = False,
    hd_percentile: int = 95,
    save_csv: Optional[str] = None,
) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    """
    Compute per-case Dice, HD, HDp (p=hd_percentile) across two folders of masks and
    return (rows, summary). Optionally save rows to CSV.

    Returns:
      rows:   [{'id': str, 'dice': float, 'hd': float, 'hdp': float}, ...]
      summary:{'dice_mean': float, 'dice_std': float, 'hd_mean': float, 'hd_std': float,
               'hdp_mean': float, 'hdp_std': float}
    """
    pairs = list_pairs(folder_a, folder_b)

    rows = []
    dice_vals, hd_vals, hdp_vals = [], [], []

    for fn, pa, pb in pairs:
        try:
            d, h, hp = compute_metrics_for_pair(
                gt_path=pa,
                pred_path=pb,
                include_background=include_background,
                hd_percentile=hd_percentile
            )
            case_id = os.path.splitext(os.path.splitext(fn)[0])[0]  # strip .nii(.gz)
            rows.append({"id": case_id, "dice": d, "hd": h, "hdp": hp})
            dice_vals.append(d); hd_vals.append(h); hdp_vals.append(hp)
            print(f"{fn}: Dice={d:.4f}  HD={h:.2f} mm  HD{hd_percentile}={hp:.2f} mm")
        except Exception as e:
            print(f" {fn}: error -> {e}")

    # Summary
    dm, ds = mean_std(dice_vals)
    hm, hs = mean_std(hd_vals)
    hpm, hps = mean_std(hdp_vals)

    summary = {
        "dice_mean": dm, "dice_std": ds,
        "hd_mean": hm, "hd_std": hs,
        "hdp_mean": hpm, "hdp_std": hps,
        "hd_percentile": hd_percentile,
        "include_background": include_background,
        "n": len(rows),
    }

    if save_csv and rows:
        with open(save_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id", "dice", "hd", "hdp"])
            w.writeheader()
            w.writerows(rows)
        print(f"Saved per-case CSV to: {save_csv}")

    return rows, summary
