import SimpleITK as sitk
import numpy as np

# ============================================================

# UTILITAIRES

# ============================================================

def voxel_volume_ml(img):
"""
Compute voxel volume in mL from image spacing.
PET/CT spacing is in mm → convert mm^3 to mL.
"""
sx, sy, sz = img.GetSpacing()
return (sx * sy * sz) / 1000.0

def resample_to_pet(ct_img, pet_img):
"""
Resample CT image (or mask) into PET space.
This is REQUIRED because PET and CT often have different resolutions.
"""
resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(pet_img)
resampler.SetInterpolator(sitk.sitkNearestNeighbor)
resampler.SetTransform(sitk.Transform())
resampler.SetDefaultPixelValue(0)
return resampler.Execute(ct_img)

# ============================================================

# CT-BASED FILTERING

# ============================================================

def create_soft_tissue_mask(ct_img):
"""
Create a rough soft-tissue mask using CT intensities (HU).

```
Typical HU ranges:
- Air:        < -500
- Lung:       ~ -700 to -300
- Fat:        ~ -200 to -50
- Soft tissue: ~ -50 to 150
- Bone:       > 300

We keep soft tissue + fat (depending on your use case).
"""
ct_np = sitk.GetArrayFromImage(ct_img)

# Keep voxels in a reasonable biological range
mask = (ct_np > -200) & (ct_np < 300)

out = sitk.GetImageFromArray(mask.astype(np.uint8))
out.CopyInformation(ct_img)
return out
```

def remove_high_density_regions(ct_img):
"""
Identify very dense structures (bone, calcifications).

```
We will later REMOVE PET detections overlapping these regions.
"""
ct_np = sitk.GetArrayFromImage(ct_img)

# Bone threshold (approximate)
bone_mask = ct_np > 300

out = sitk.GetImageFromArray(bone_mask.astype(np.uint8))
out.CopyInformation(ct_img)
return out
```

# ============================================================

# PET SEGMENTATION (IMPROVED)

# ============================================================

def pet_local_peak_segmentation(pet_img, roi_mask, rel_thr=0.45, seed_frac=0.3):
"""
Core idea, but isolated and reusable.

```
Steps:
1. Restrict PET to ROI
2. Seed regions using a low threshold (catch cold lesions)
3. Split into connected components
4. For each component → adaptive threshold using local peak
"""

pet_np = sitk.GetArrayFromImage(pet_img)
roi_np = sitk.GetArrayFromImage(roi_mask)

# Apply ROI mask
pet_roi = np.where(roi_np > 0, pet_np, 0.0)

positive = pet_roi[pet_roi > 0]
if positive.size == 0:
    return sitk.Image(pet_img.GetSize(), sitk.sitkUInt8)

global_max = float(positive.max())

# Step 1: seed threshold (low)
seed_thr = seed_frac * global_max
seed = (pet_roi >= seed_thr).astype(np.uint8)

seed_img = sitk.GetImageFromArray(seed)
seed_img.CopyInformation(pet_img)

# Step 2: connected components
cc = sitk.ConnectedComponent(seed_img)
cc_np = sitk.GetArrayFromImage(cc)

labels = np.unique(cc_np)
labels = labels[labels != 0]

final = np.zeros_like(cc_np, dtype=np.uint8)

# Step 3: local refinement
for lab in labels:
    region = (cc_np == lab)

    local_peak = float(pet_roi[region].max())
    local_thr = rel_thr * local_peak

    refined = (pet_roi >= local_thr) & region
    final |= refined.astype(np.uint8)

out = sitk.GetImageFromArray(final)
out.CopyInformation(pet_img)
return out
```

# ============================================================

# POST-PROCESSING

# ============================================================

def remove_small_components(mask_img, min_volume_ml):
"""
Remove small noisy detections.
"""
cc = sitk.ConnectedComponent(mask_img)
stats = sitk.LabelShapeStatisticsImageFilter()
stats.Execute(cc)

```
vx = voxel_volume_ml(mask_img)

cc_np = sitk.GetArrayFromImage(cc)
keep = np.zeros_like(cc_np, dtype=np.uint8)

for lab in stats.GetLabels():
    vol = stats.GetNumberOfPixels(lab) * vx
    if vol >= min_volume_ml:
        keep[cc_np == lab] = 1

out = sitk.GetImageFromArray(keep)
out.CopyInformation(mask_img)
return out
```

def apply_ct_constraints(tumor_mask, soft_mask, bone_mask):
"""
Apply anatomical constraints from CT.

```
- Remove detections in bone
- Keep only plausible tissue regions
"""

tumor_np = sitk.GetArrayFromImage(tumor_mask)
soft_np = sitk.GetArrayFromImage(soft_mask)
bone_np = sitk.GetArrayFromImage(bone_mask)

# Remove bone overlap
tumor_np[bone_np > 0] = 0

# Keep only soft tissue
tumor_np = tumor_np & (soft_np > 0)

out = sitk.GetImageFromArray(tumor_np.astype(np.uint8))
out.CopyInformation(tumor_mask)
return out
```

# ============================================================

# MAIN PIPELINE

# ============================================================

def advanced_pet_ct_pipeline(pet_img, ct_img, breast_mask,
rel_thr=0.45,
seed_frac=0.3,
min_volume_ml=0.5):
"""
FULL PIPELINE WITHOUT ML

```
Steps:
1. Resample CT + breast mask to PET space
2. Build CT-derived masks (soft tissue, bone)
3. PET segmentation (local peak method)
4. Apply CT constraints
5. Remove small components
6. Return final tumor mask
"""

# --------------------------------------------------------
# Step 1: align everything to PET
# --------------------------------------------------------
ct_pet = resample_to_pet(ct_img, pet_img)
breast_pet = resample_to_pet(breast_mask, pet_img)

# --------------------------------------------------------
# Step 2: CT-derived masks
# --------------------------------------------------------
soft_mask = create_soft_tissue_mask(ct_pet)
bone_mask = remove_high_density_regions(ct_pet)

# Restrict soft mask to breast ROI
soft_np = sitk.GetArrayFromImage(soft_mask)
breast_np = sitk.GetArrayFromImage(breast_pet)
soft_np = soft_np & (breast_np > 0)

soft_mask = sitk.GetImageFromArray(soft_np.astype(np.uint8))
soft_mask.CopyInformation(ct_pet)

# --------------------------------------------------------
# Step 3: PET segmentation
# --------------------------------------------------------
tumor_mask = pet_local_peak_segmentation(
    pet_img,
    breast_pet,
    rel_thr=rel_thr,
    seed_frac=seed_frac
)

# --------------------------------------------------------
# Step 4: CT constraints
# --------------------------------------------------------
tumor_mask = apply_ct_constraints(
    tumor_mask,
    soft_mask,
    bone_mask
)

# --------------------------------------------------------
# Step 5: remove small regions
# --------------------------------------------------------
tumor_mask = remove_small_components(
    tumor_mask,
    min_volume_ml
)

return tumor_mask
```
