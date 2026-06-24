import os
import glob
import numpy as np
from PIL import Image
from pathlib import Path

import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description="Preprocess Landsat satellite data into aligned tiles.")
parser.add_argument("--input_dir", type=str, default="./Guwahati_Azara_dataset", help="Path to raw satellite files")
parser.add_argument("--output_dir", type=str, default="./datasets/guwahati_azara_processed", help="Path to save processed tiles")
parser.add_argument("--tile_size", type=int, default=256, help="Size of cropped tiles")
parser.add_argument("--stride", type=int, default=192, help="Stride distance between adjacent tiles")
parser.add_argument("--max_nodata_ratio", type=float, default=0.15, help="Maximum allowed ratio of zero values per tile")
args = parser.parse_args()

input_dir = args.input_dir
output_dir = args.output_dir
tile_size = args.tile_size
stride = args.stride
max_nodata_ratio = args.max_nodata_ratio

# Create output directories
for split in ["train", "val"]:
    os.makedirs(os.path.join(output_dir, split), exist_ok=True)

# 1. Identify all unique scenes in the dataset folder by looking at files ending in _SR_B2.TIF
all_b2_files = glob.glob(os.path.join(input_dir, "*_SR_B2.TIF"))
scene_prefixes = sorted([os.path.basename(f).replace("_SR_B2.TIF", "") for f in all_b2_files])

print(f"Found {len(scene_prefixes)} unique Landsat scenes:")
for pref in scene_prefixes:
    print(f"  {pref}")

# Split scenes: use first 7 scenes for training, and last 2 for validation
train_prefixes = scene_prefixes[:7]
val_prefixes = scene_prefixes[7:]

print(f"\nTraining scenes ({len(train_prefixes)}):")
for p in train_prefixes:
    print(f"  {p}")

print(f"Validation scenes ({len(val_prefixes)}):")
for p in val_prefixes:
    print(f"  {p}")

def process_scene(prefix, split):
    print(f"\nProcessing scene: {prefix} ({split})")
    
    b2_path = os.path.join(input_dir, f"{prefix}_SR_B2.TIF")
    b3_path = os.path.join(input_dir, f"{prefix}_SR_B3.TIF")
    b4_path = os.path.join(input_dir, f"{prefix}_SR_B4.TIF")
    b10_path = os.path.join(input_dir, f"{prefix}_ST_B10.TIF")
    
    # Verify all bands exist
    for p in [b2_path, b3_path, b4_path, b10_path]:
        if not os.path.exists(p):
            print(f"Error: Missing band file {p}")
            return 0
            
    # Load bands using Pillow
    print("  Loading bands into memory...")
    r_arr = np.array(Image.open(b4_path), dtype=np.float32)
    g_arr = np.array(Image.open(b3_path), dtype=np.float32)
    b_arr = np.array(Image.open(b2_path), dtype=np.float32)
    ir_arr = np.array(Image.open(b10_path), dtype=np.float32)
    
    # 2. Landsat 8/9 scientific normalization
    print("  Normalizing visible bands (RGB)...")
    r_refl = r_arr * 0.0000275 - 0.2
    g_refl = g_arr * 0.0000275 - 0.2
    b_refl = b_arr * 0.0000275 - 0.2
    
    r_norm = np.clip(r_refl, 0.0, 0.3) / 0.3 * 255.0
    g_norm = np.clip(g_refl, 0.0, 0.3) / 0.3 * 255.0
    b_norm = np.clip(b_refl, 0.0, 0.3) / 0.3 * 255.0
    
    rgb = np.stack([r_norm, g_norm, b_norm], axis=2).astype(np.uint8)
    
    # Surface Temperature scaling
    print("  Normalizing thermal IR band...")
    temp_K = ir_arr * 0.00341802 + 149.0
    temp_norm = np.clip(temp_K, 280.0, 315.0)
    temp_norm = (temp_norm - 280.0) / (315.0 - 280.0) * 255.0
    ir = temp_norm.astype(np.uint8)
    
    # 3. Tiling
    H, W = ir.shape
    tile_count = 0
    skipped_count = 0
    
    print(f"  Tiling image of size {H}x{W}...")
    for y in range(0, H - tile_size, stride):
        for x in range(0, W - tile_size, stride):
            # Extract patches
            patch_rgb = rgb[y:y+tile_size, x:x+tile_size]
            patch_ir = ir[y:y+tile_size, x:x+tile_size]
            
            raw_ir_patch = ir_arr[y:y+tile_size, x:x+tile_size]
            nodata_ratio = np.mean(raw_ir_patch == 0)
            
            if nodata_ratio > max_nodata_ratio:
                skipped_count += 1
                continue
                
            patch_ir_3ch = np.stack([patch_ir, patch_ir, patch_ir], axis=2)
            combined_tile = np.concatenate([patch_ir_3ch, patch_rgb], axis=1)
            
            tile_name = f"{prefix}_y{y}_x{x}.png"
            tile_path = os.path.join(output_dir, split, tile_name)
            Image.fromarray(combined_tile).save(tile_path)
            tile_count += 1
            
    print(f"  Completed! Saved {tile_count} tiles (Skipped {skipped_count} nodata tiles)")
    return tile_count

# Process training scenes
total_train = 0
for prefix in train_prefixes:
    total_train += process_scene(prefix, "train")

# Process validation scenes
total_val = 0
for prefix in val_prefixes:
    total_val += process_scene(prefix, "val")

print("\n==========================================")
print(f"Dataset Preprocessing Complete!")
print(f"Saved to: {output_dir}")
print(f"Total training tiles: {total_train}")
print(f"Total validation tiles: {total_val}")
print("==========================================")
