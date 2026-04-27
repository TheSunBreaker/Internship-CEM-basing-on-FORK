"""
features_pipeline.py (FAST + Compact)
- 2 mm isotropic by default, align CT+breast+tumor to PET reference
- NumPy tumor features (PET & CT) + tumor shape (compact)
- PyRadiomics (small subset) ONLY on the 0–5 mm ring for PET & CT
"""

from __future__ import annotations
import os, glob, math
import numpy as np
import pandas as pd
import nibabel as nib
from nibabel import processing as nibproc
from typing import Dict, Tuple, Optional, List
from scipy.ndimage import label, binary_dilation, generate_binary_structure

# --------------------------------
# Speed defaults
# --------------------------------
ISO_SPACING = (2.0, 2.0, 2.0)
DILATION_NEIGHBORHOOD = 1  # 6-connected

# --------------------------
# I/O / resampling
# --------------------------

def load_nifti(path: str):
    img = nib.load(path)
    arr = img.get_fdata(dtype=np.float32)
    sp = tuple(float(x) for x in img.header.get_zooms()[:3])
    return img, arr, sp

def _to_RAS(img: nib.Nifti1Image) -> nib.Nifti1Image:
    return nib.as_closest_canonical(img)

def _resample_to_output(img: nib.Nifti1Image, spacing=(1.0,1.0,1.0), is_mask=False) -> nib.Nifti1Image:
    order = 0 if is_mask else 1
    return nibproc.resample_to_output(img, voxel_sizes=spacing, order=order)

def _resample_from_to(ref: nib.Nifti1Image, moving: nib.Nifti1Image, is_mask=False) -> nib.Nifti1Image:
    order = 0 if is_mask else 1
    return nibproc.resample_from_to(moving, ref, order=order)

def isotropic_and_align_to_pet(
    pet_img: nib.Nifti1Image,
    ct_img: nib.Nifti1Image,
    breast_mask_img: nib.Nifti1Image,
    tumor_mask_img: nib.Nifti1Image,
    iso=ISO_SPACING,
):
    # 1) RAS+
    pet_r, ct_r = _to_RAS(pet_img), _to_RAS(ct_img)
    br_r,  tu_r = _to_RAS(breast_mask_img), _to_RAS(tumor_mask_img)

    # 2) Each -> isotropic
    pet_iso = _resample_to_output(pet_r, spacing=iso, is_mask=False)
    ct_iso  = _resample_to_output(ct_r,  spacing=iso, is_mask=False)
    br_iso  = _resample_to_output(br_r,  spacing=iso, is_mask=True)
    tu_iso  = _resample_to_output(tu_r,  spacing=iso, is_mask=True)

    # 3) Align to PET isotropic grid
    ct_on_pet = _resample_from_to(pet_iso, ct_iso, is_mask=False)
    br_on_pet = _resample_from_to(pet_iso, br_iso, is_mask=True)
    tu_on_pet = _resample_from_to(pet_iso, tu_iso, is_mask=True)

    pet = pet_iso.get_fdata(dtype=np.float32)
    ct  = ct_on_pet.get_fdata(dtype=np.float32)
    breast = (br_on_pet.get_fdata(dtype=np.float32) > 0.5)
    tumor  = (tu_on_pet.get_fdata(dtype=np.float32) > 0.5)

    assert pet.shape == ct.shape == breast.shape == tumor.shape, "Aligned shapes differ"
    spacing = tuple(float(x) for x in pet_iso.header.get_zooms()[:3])
    return pet, ct, breast, tumor, spacing

# --------------------------
# Utils / ROI cropping
# --------------------------

def _crop_to_mask_bbox(arrs: List[np.ndarray], mask: np.ndarray, spacing, margin_mm=12.0):
    if mask.sum() == 0:
        return arrs, (slice(0,arrs[0].shape[0]), slice(0,arrs[0].shape[1]), slice(0,arrs[0].shape[2]))
    idx = np.argwhere(mask)
    mins = idx.min(axis=0)
    maxs = idx.max(axis=0)
    vox_margin = np.array([max(1, int(round(margin_mm / s))) for s in spacing], dtype=int)
    lo = np.maximum(mins - vox_margin, 0)
    hi = np.minimum(maxs + vox_margin + 1, mask.shape)
    slicer = tuple(slice(lo[d], hi[d]) for d in range(3))
    return [a[slicer] for a in arrs], slicer

# --------------------------
# Regions / rings
# --------------------------

def _ring_by_dilation(tumor_mask: np.ndarray,
                      breast_mask: np.ndarray,
                      spacing: Tuple[float,float,float],
                      inner_mm: float,
                      outer_mm: float) -> np.ndarray:
    if tumor_mask.sum() == 0 or breast_mask.sum() == 0:
        return breast_mask & False
    vox_inner = max(0, int(round(inner_mm / spacing[0])))
    vox_outer = max(vox_inner+1, int(round(outer_mm / spacing[0])))
    struct = generate_binary_structure(3, DILATION_NEIGHBORHOOD)
    inner = binary_dilation(tumor_mask, structure=struct, iterations=vox_inner) if vox_inner > 0 else tumor_mask
    outer = binary_dilation(tumor_mask, structure=struct, iterations=vox_outer)
    ring = (outer & (~inner)) & breast_mask
    ring &= (~tumor_mask)
    return ring

def split_breasts(breast_mask: np.ndarray, tumor_mask: np.ndarray):
    labeled, n = label(breast_mask.astype(np.uint8))
    if n >= 2:
        overlaps = [np.logical_and(labeled == (i+1), tumor_mask).sum() for i in range(n)]
        ipsi_label = int(np.argmax(overlaps)) + 1
        ipsi = labeled == ipsi_label
        contra = (labeled != ipsi_label) & (labeled > 0)
        return ipsi, contra

    coords = np.argwhere(breast_mask)
    if coords.size == 0:
        return breast_mask & False, breast_mask & False
    minc = coords.min(axis=0); maxc = coords.max(axis=0)
    axis = int(np.argmax(maxc - minc))
    mid = (minc[axis] + maxc[axis]) / 2.0
    tcoords = np.argwhere(tumor_mask)
    if tcoords.size == 0:
        ipsi = breast_mask & (np.indices(breast_mask.shape)[axis] < mid)
        return ipsi, breast_mask & (~ipsi)
    tcent = tcoords.mean(axis=0)[axis]
    ipsi = breast_mask & ((np.indices(breast_mask.shape)[axis] < mid) if tcent < mid else (np.indices(breast_mask.shape)[axis] >= mid))
    return ipsi, breast_mask & (~ipsi)

# --------------------------
# NumPy features (compact)
# --------------------------

def voxel_volume_mm3(spacing): return float(spacing[0]*spacing[1]*spacing[2])

def first_order(arr: np.ndarray, mask: np.ndarray, prefix: str) -> Dict[str, float]:
    vals = arr[mask]
    if vals.size == 0:
        return {f"{prefix}_n": 0, f"{prefix}_mean": np.nan, f"{prefix}_std": np.nan,
                f"{prefix}_min": np.nan, f"{prefix}_p10": np.nan, f"{prefix}_median": np.nan,
                f"{prefix}_p90": np.nan, f"{prefix}_max": np.nan}
    return {f"{prefix}_n": int(vals.size),
            f"{prefix}_mean": float(vals.mean()),
            f"{prefix}_std": float(vals.std(ddof=1)) if vals.size>1 else 0.0,
            f"{prefix}_min": float(vals.min()),
            f"{prefix}_p10": float(np.percentile(vals,10)),
            f"{prefix}_median": float(np.median(vals)),
            f"{prefix}_p90": float(np.percentile(vals,90)),
            f"{prefix}_max": float(vals.max())}

def _surface_area_voxel(mask: np.ndarray, spacing: Tuple[float,float,float]) -> float:
    sx, sy, sz = spacing; area = 0.0
    a01 = (~mask[1:,:,:]) & mask[:-1,:,:]; a10 = (~mask[:-1,:,:]) & mask[1:,:,:]; area += (a01.sum()+a10.sum())*(sy*sz)
    b01 = (~mask[:,1:,:]) & mask[:,:-1,:]; b10 = (~mask[:,:-1,:]) & mask[:,1:,:]; area += (b01.sum()+b10.sum())*(sx*sz)
    c01 = (~mask[:,:,1:]) & mask[:,:,:-1]; c10 = (~mask[:,:,:-1]) & mask[:,:,1:]; area += (c01.sum()+c10.sum())*(sx*sy)
    return float(area)

def shape_features(mask: np.ndarray, spacing: Tuple[float,float,float], prefix: str) -> Dict[str, float]:
    vv = voxel_volume_mm3(spacing); voxels = int(mask.sum()); vol_mm3 = voxels * vv; vol_ml = vol_mm3 / 1000.0
    if voxels == 0:
        return {f"{prefix}_voxels":0, f"{prefix}_volume_ml":0.0, f"{prefix}_surface_mm2":np.nan, f"{prefix}_sphericity":np.nan,
                f"{prefix}_bbox_x_mm":np.nan, f"{prefix}_bbox_y_mm":np.nan, f"{prefix}_bbox_z_mm":np.nan,
                f"{prefix}_bbox_volume_ml":np.nan, f"{prefix}_compactness_bbox":np.nan}
    coords = np.argwhere(mask); minc = coords.min(axis=0); maxc = coords.max(axis=0)
    dims_vox = (maxc - minc + 1).astype(np.float32); dims_mm = dims_vox * np.array(spacing, dtype=np.float32)
    surface = _surface_area_voxel(mask, spacing)
    sphericity = ((math.pi ** (1/3.0)) * (6.0 * vol_mm3) ** (2/3.0)) / surface if surface > 0 else np.nan
    bbox_vol_ml = float(np.prod(dims_mm) / 1000.0); compact_bbox = (vol_ml / bbox_vol_ml) if bbox_vol_ml>0 else np.nan
    return {f"{prefix}_voxels":voxels, f"{prefix}_volume_ml":float(vol_ml), f"{prefix}_surface_mm2":float(surface),
            f"{prefix}_sphericity":float(sphericity), f"{prefix}_bbox_x_mm":float(dims_mm[0]), f"{prefix}_bbox_y_mm":float(dims_mm[1]),
            f"{prefix}_bbox_z_mm":float(dims_mm[2]), f"{prefix}_bbox_volume_ml":float(bbox_vol_ml), f"{prefix}_compactness_bbox":float(compact_bbox)}

def suv_peak_3x3x3(pet: np.ndarray, tumor_mask: np.ndarray) -> float:
    if tumor_mask.sum() == 0: return np.nan
    t = pet.copy(); t[~tumor_mask] = -np.inf
    x,y,z = np.unravel_index(int(np.argmax(t)), t.shape)
    xs = slice(max(0,x-1), min(pet.shape[0], x+2))
    ys = slice(max(0,y-1), min(pet.shape[1], y+2))
    zs = slice(max(0,z-1), min(pet.shape[2], z+2))
    vals = pet[xs,ys,zs][tumor_mask[xs,ys,zs]]
    return float(vals.mean()) if vals.size else np.nan

def mtv_tlg(pet: np.ndarray, tumor_mask: np.ndarray, spacing: Tuple[float,float,float], mode: str="41pct"):
    vals = pet[tumor_mask]
    if vals.size == 0: return 0.0, 0.0
    thr = float(vals.max()*0.41) if mode=="41pct" else 2.5
    sub = tumor_mask & (pet >= thr)
    if sub.sum()==0: return 0.0, 0.0
    vol_ml = sub.sum() * voxel_volume_mm3(spacing) / 1000.0
    tlg = float(pet[sub].mean() * vol_ml)
    return float(vol_ml), float(tlg)

def asymmetry_metric(ipsi_vals: np.ndarray, contra_vals: np.ndarray) -> float:
    if ipsi_vals.size==0 or contra_vals.size==0: return np.nan
    denom = float(contra_vals.mean())
    return float(ipsi_vals.mean()/denom) if denom!=0 else np.nan

# --------------------------
# Minimal PyRadiomics on ring 0–5 mm
# --------------------------

def _np_to_sitk(img_np: np.ndarray, spacing: Tuple[float,float,float], is_mask: bool):
    import SimpleITK as sitk
    im = sitk.GetImageFromArray((img_np.astype(np.uint8) if is_mask else img_np.astype(np.float32)))
    im.SetSpacing(tuple(float(s) for s in spacing))
    return im

def _make_basic_ring_extractor(bin_width: float, glcm_distances: List[int] = [1]):
    try:
        from radiomics import featureextractor
    except Exception as e:
        raise RuntimeError("PyRadiomics import failed (ring features will be skipped): " + str(e))
    settings = {
        "binWidth": float(bin_width),
        "normalize": False,
        "resampledPixelSpacing": None,
        "interpolator": "sitkBSpline",
        "label": 1,
        "preCrop": True,
        "correctMask": True,
        "geometryTolerance": 1e-6,
        "distances": glcm_distances,
    }
    extr = featureextractor.RadiomicsFeatureExtractor(**settings)
    extr.enableImageTypes(Original={})
    extr.disableAllFeatures()
    # Keep just a few first-order + a few GLCM
    extr.enableFeaturesByName(firstorder=[
        "Mean","Median","Minimum","Maximum","10Percentile","90Percentile",
        "Variance","Skewness","Kurtosis","Entropy"
    ])
    extr.enableFeaturesByName(glcm=[
        "Contrast","Correlation","JointEnergy","Id","Idm","JointEntropy"
    ])
    return extr

def _pyrad_ring_features(extr, image_np: np.ndarray, ring_mask: np.ndarray, spacing, prefix: str) -> Dict[str, float]:
    if ring_mask.sum() == 0:
        return {f"{prefix}EMPTY": 1}
    img_sitk = _np_to_sitk(image_np, spacing, is_mask=False)
    msk_sitk = _np_to_sitk(ring_mask.astype(np.uint8), spacing, is_mask=True)
    result = extr.execute(img_sitk, msk_sitk, label=1)
    out: Dict[str, float] = {}
    for k, v in result.items():
        if str(k).startswith("diagnostics_"):
            continue
        try:
            out[f"{prefix}{k}"] = float(v)
        except Exception:
            try:
                out[f"{prefix}{k}"] = float(np.asarray(v).item())
            except Exception:
                pass
    return out

# --------------------------
# Case -> feature row (compact set)
# --------------------------

def case_features(case_id: str,
                  pet_img: nib.Nifti1Image,
                  ct_img: nib.Nifti1Image,
                  breast_mask_img: nib.Nifti1Image,
                  tumor_mask_img: nib.Nifti1Image,
                  ring_mm_1: float = 5.0,     # ring outer radius
                  ring_mm_2: float = 10.0,    # ignored now (we only use 0–5 mm)
                  enable_pyradiomics: bool = True,   # minimal ring-only
                  pet_binwidth_suv: float = 0.5,     # coarser bins -> faster
                  ct_binwidth_hu: float = 50.0,      # coarser bins -> faster
                  enable_log: bool = False,          # unused
                  enable_wavelet: bool = False) -> Dict[str, float]:  # unused
    # 1) Isotropic (2mm) + align to PET grid
    pet, ct, breast, tumor, spacing = isotropic_and_align_to_pet(
        pet_img, ct_img, breast_mask_img, tumor_mask_img, iso=ISO_SPACING
    )

    # 2) Crop to breast bbox (+ margin)
    [pet, ct, breast, tumor], _ = _crop_to_mask_bbox([pet, ct, breast, tumor], breast, spacing, margin_mm=12.0)

    # Keep tumor inside breast
    tumor = tumor & breast

    # CT stabilization
    ct = np.clip(ct, -150.0, 250.0).astype(np.float32)
    pet = pet.astype(np.float32)

    # Regions
    ipsi, contra = split_breasts(breast, tumor)
    ring05 = _ring_by_dilation(tumor, breast, spacing, 0.0, ring_mm_1)
    breast_bg = breast & (~tumor)
    contra_bg = contra & (~tumor)

    feats: Dict[str, float] = {"case_id": case_id}

    # ---- Tumor SHAPE (compact ~8)
    feats.update(shape_features(tumor, spacing, prefix="tumor_shape"))

    # ---- PET Tumor (compact ~9 + 4 MTV/TLG + 1 TBR = ~14)
    fo_pet_tumor = first_order(pet, tumor, "pet_tumor")
    keep_pet = ["pet_tumor_mean","pet_tumor_std","pet_tumor_min","pet_tumor_median","pet_tumor_max","pet_tumor_p10","pet_tumor_p90"]
    feats.update({k: fo_pet_tumor[k] for k in keep_pet})
    feats["pet_tumor_SUVpeak3x3x3"] = suv_peak_3x3x3(pet, tumor)
    mtv41, tlg41 = mtv_tlg(pet, tumor, spacing, "41pct"); feats["pet_tumor_MTV41_ml"] = mtv41; feats["pet_tumor_TLG41"] = tlg41
    mtv25, tlg25 = mtv_tlg(pet, tumor, spacing, "2.5");  feats["pet_tumor_MTV2p5_ml"]  = mtv25; feats["pet_tumor_TLG2p5"]  = tlg25
    # simple TBR
    cmean = float(pet[contra_bg].mean()) if contra_bg.any() else np.nan
    tmean = feats["pet_tumor_mean"]
    feats["pet_TBR_tumorMean_over_contraMean"] = (tmean / cmean) if (cmean and not np.isnan(cmean) and cmean!=0) else np.nan

    # ---- CT Tumor (compact ~7)
    fo_ct_tumor = first_order(ct, tumor, "ct_tumor_HU")
    keep_ct = ["ct_tumor_HU_mean","ct_tumor_HU_std","ct_tumor_HU_min","ct_tumor_HU_median","ct_tumor_HU_max","ct_tumor_HU_p10","ct_tumor_HU_p90"]
    feats.update({k: fo_ct_tumor[k] for k in keep_ct})

    # ---- Minimal PyRadiomics on ring 0–5 mm for PET & CT (~16 × 2 = ~32)
    if enable_pyradiomics:
        try:
            pet_extr = _make_basic_ring_extractor(pet_binwidth_suv, glcm_distances=[1])
            ct_extr  = _make_basic_ring_extractor(ct_binwidth_hu,  glcm_distances=[1])
            feats.update(_pyrad_ring_features(pet_extr, pet, ring05, spacing, prefix="pyrad_pet_ring0to5__"))
            feats.update(_pyrad_ring_features(ct_extr,  ct,  ring05, spacing, prefix="pyrad_ct_ring0to5__"))
        except Exception as e:
            feats["pyradiomics_error"] = 1
            feats["pyradiomics_error_msg"] = str(e)

    # meta
    feats["ring_inner_mm"] = 0.0
    feats["ring_outer_mm"] = float(ring_mm_1)
    feats["isotropic_spacing_mm"] = float(spacing[0])
    feats["pyradiomics_enabled"] = int(enable_pyradiomics)

    return feats

# --------------------------
# Subject discovery / saving
# --------------------------

def _match_file(patterns: List[str]) -> Optional[str]:
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if hits: return hits[0]
    return None

def discover_subjects(route_root: str,
                      breast_masks_dir: str,
                      tumor_masks_dir: str,
                      ct_suffix: str = "_TDM",
                      pet_suv_suffix: str = "_TEP_SUV") -> List[Dict[str, str]]:
    subs = [d for d in sorted(os.listdir(route_root)) if os.path.isdir(os.path.join(route_root, d))]
    cases: List[Dict[str, str]] = []
    for sub in subs:
        subdir = os.path.join(route_root, sub); subj_id = sub
        ct_path = _match_file([os.path.join(subdir, f"{subj_id}{ct_suffix}.nii.gz"),
                               os.path.join(subdir, f"{subj_id}{ct_suffix}.nii")])
        pet_path = _match_file([os.path.join(subdir, f"{subj_id}{pet_suv_suffix}.nii.gz"),
                                os.path.join(subdir, f"{subj_id}{pet_suv_suffix}.nii")])
        if ct_path is None or pet_path is None:
            print(f"[WARN] {subj_id}: missing CT or PET_SUV; skipping."); continue
        breast_path = _match_file([os.path.join(breast_masks_dir, f"{subj_id}_TDM_breast_mask.nii.gz"),
                                   os.path.join(breast_masks_dir, f"{subj_id}_TDM_breast_mask.nii")])
        tumor_path = _match_file([os.path.join(tumor_masks_dir, f"{subj_id}_tumor_mask.nii"),
                                  os.path.join(tumor_masks_dir, f"{subj_id}_tumor_mask.nii.gz")])
        if breast_path is None or tumor_path is None:
            print(f"[WARN] {subj_id}: missing masks; skipping."); continue
        cases.append({"case_id": subj_id, "ct": ct_path, "pet": pet_path, "breast": breast_path, "tumor": tumor_path})
    return cases

def save_dataset(rows: List[Dict[str, float]], out_csv: str, out_xlsx: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if "case_id" in df.columns:
        cols = ["case_id"] + [c for c in df.columns if c != "case_id"]
        df = df[cols]
    df.to_csv(out_csv, index=False)
    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="features")
    return df
