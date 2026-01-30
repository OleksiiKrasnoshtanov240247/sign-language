"""
Complete Pipeline: Extract, Augment, Replace Problem Letters + Add Nonsense Class
==================================================================================
1. Extract landmarks from photos (MediaPipe)
2. Augment x10 (noise, scale, rotate, translate)
3. Normalize (center wrist, scale by point 9)
4. Remove old D, F, G, S, V from ngt_data.csv
5. Add new augmented data (replaced letters + Nonsense class)
6. Save as ngt_final.csv with 25 classes (24 letters + Nonsense)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ============ CONFIGURATION ============

MEDIAPIPE_MODEL = "hand_landmarker.task"
PHOTOS_DIR = Path("letters")
INPUT_CSV = "ngt_data.csv"
OUTPUT_CSV = "ngt_final.csv"

LETTERS_TO_REPLACE = ['D', 'F', 'G', 'S', 'V', 'X']  # Replace these with your photos
CLASSES_TO_ADD = ['Nonsense']  # Add these as new classes (25th class)
AUGMENT_MULTIPLIER = 10

# ============ EXTRACTION (from frankenstein_builder.py) ============

def extract_landmarks_from_image(detector, img_path):
    """Extract 21 landmarks (63 coordinates) from image using MediaPipe"""
    mp_image = mp.Image.create_from_file(str(img_path))
    results = detector.detect(mp_image)
    
    if results.hand_landmarks and len(results.hand_landmarks) > 0:
        hand = results.hand_landmarks[0]
        coords = []
        for lm in hand:
            coords.extend([lm.x, lm.y, lm.z])
        return np.array(coords, dtype=np.float32)
    return None

def extract_letter_landmarks(detector, letter, photos_dir):
    """Extract all landmarks for given letter from photo folder"""
    # Try both naming patterns
    letter_folder = photos_dir / f"{letter}_letter"
    if not letter_folder.exists():
        letter_folder = photos_dir / letter
    
    if not letter_folder.exists():
        print(f"   ✗ Folder not found: {letter_folder}")
        return np.array([])
    
    image_files = list(letter_folder.glob("*.jpg")) + \
                  list(letter_folder.glob("*.png")) + \
                  list(letter_folder.glob("*.jpeg")) + \
                  list(letter_folder.glob("*.JPG")) + \
                  list(letter_folder.glob("*.PNG"))
    
    if not image_files:
        print(f"   ✗ No images found in: {letter_folder}")
        return np.array([])
    
    landmarks_list = []
    failed = 0
    
    for img_path in tqdm(image_files, desc=f"   Extracting {letter}", leave=False):
        landmarks = extract_landmarks_from_image(detector, img_path)
        if landmarks is not None:
            landmarks_list.append(landmarks)
        else:
            failed += 1
    
    print(f"   ✓ {letter}: {len(landmarks_list)} extracted, {failed} failed")
    return np.array(landmarks_list) if landmarks_list else np.array([])

# ============ AUGMENTATION (from augment_landmarks.py) ============

def add_noise(coords, std=0.01):
    """Add random Gaussian noise"""
    return coords + np.random.normal(0, std, coords.shape)

def scale(coords, factor_range=(0.85, 1.15)):
    """Scale landmarks around center"""
    factor = np.random.uniform(*factor_range)
    points = coords.reshape(-1, 3)
    center = points.mean(axis=0)
    return ((points - center) * factor + center).flatten()

def rotate_2d(coords, max_angle=20):
    """Rotate landmarks in XY plane"""
    angle = np.radians(np.random.uniform(-max_angle, max_angle))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    points = coords.reshape(-1, 3)
    x, y = points[:, 0], points[:, 1]
    cx, cy = x.mean(), y.mean()
    x_new = (x - cx) * cos_a - (y - cy) * sin_a + cx
    y_new = (x - cx) * sin_a + (y - cy) * cos_a + cy
    points[:, 0], points[:, 1] = x_new, y_new
    return points.flatten()

def translate(coords, max_shift=0.08):
    """Translate landmarks randomly"""
    points = coords.reshape(-1, 3)
    points[:, 0] += np.random.uniform(-max_shift, max_shift)
    points[:, 1] += np.random.uniform(-max_shift, max_shift)
    return points.flatten()

def augment_sample(coords):
    """Apply random augmentations to single sample"""
    aug = coords.copy()
    aug = add_noise(aug, std=np.random.uniform(0.005, 0.015))
    
    if np.random.random() < 0.7:
        aug = scale(aug)
    if np.random.random() < 0.7:
        aug = rotate_2d(aug)
    if np.random.random() < 0.5:
        aug = translate(aug)
    
    return aug

def augment_letter_data(data, multiplier):
    """Augment letter data by multiplier"""
    augmented = list(data)
    
    for _ in range(multiplier - 1):
        for sample in data:
            augmented.append(augment_sample(sample))
    
    return np.array(augmented, dtype=np.float32)

# ============ NORMALIZATION (from merge_custom_letters.py) ============

def normalize_landmarks(coords):
    """
    Normalize landmarks:
    1. Center on wrist (point 0)
    2. Scale by hand size (distance to middle finger base - point 9)
    """
    points = coords.reshape(21, 3)
    
    # Center on wrist
    wrist = points[0].copy()
    points = points - wrist
    
    # Scale by hand size (point 9 = middle finger base)
    scale_factor = np.linalg.norm(points[9])
    if scale_factor > 0.001:
        points = points / scale_factor
    
    return points.flatten()

# ============ MAIN PIPELINE ============

def main():
    print("=" * 60)
    print("COMPLETE PIPELINE: REPLACE PROBLEM LETTERS")
    print("=" * 60)
    
    # Step 1: Initialize MediaPipe
    print("\n" + "=" * 60)
    print("STEP 1: INITIALIZE MEDIAPIPE")
    print("=" * 60)
    
    base_options = python.BaseOptions(model_asset_path=MEDIAPIPE_MODEL)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    detector = vision.HandLandmarker.create_from_options(options)
    print("✓ MediaPipe initialized")
    
    # Step 2: Extract landmarks from photos
    print("\n" + "=" * 60)
    print("STEP 2: EXTRACT LANDMARKS FROM PHOTOS")
    print("=" * 60)
    
    extracted_data = {}
    
    # Extract problem letters
    for letter in LETTERS_TO_REPLACE:
        landmarks = extract_letter_landmarks(detector, letter, PHOTOS_DIR)
        if len(landmarks) > 0:
            extracted_data[letter] = landmarks
    
    # Extract new classes (Nonsense, gibberish, etc.)
    for class_name in CLASSES_TO_ADD:
        print(f"\n   Extracting {class_name} class...")
        landmarks = extract_letter_landmarks(detector, class_name, PHOTOS_DIR)
        if len(landmarks) > 0:
            extracted_data[class_name] = landmarks
            print(f"   ✓ {class_name}: {len(landmarks)} samples (gibberish rejection)")
        else:
            print(f"   ⚠ No {class_name} data found - model won't reject this type!")
    
    if not extracted_data:
        print("\n✗ No data extracted! Check your photo folders.")
        return
    
    print(f"\n✓ Extracted {sum(len(d) for d in extracted_data.values())} images total")
    
    # Step 3: Augment extracted data
    print("\n" + "=" * 60)
    print(f"STEP 3: AUGMENT x{AUGMENT_MULTIPLIER}")
    print("=" * 60)
    
    augmented_data = {}
    
    for letter, data in extracted_data.items():
        print(f"   {letter}: {len(data)} → ", end="")
        augmented = augment_letter_data(data, AUGMENT_MULTIPLIER)
        augmented_data[letter] = augmented
        print(f"{len(augmented)} samples")
    
    print(f"\n✓ Total augmented samples: {sum(len(d) for d in augmented_data.values())}")
    
    # Step 4: Normalize augmented data
    print("\n" + "=" * 60)
    print("STEP 4: NORMALIZE LANDMARKS")
    print("=" * 60)
    
    normalized_X = []
    normalized_y = []
    
    for letter, data in augmented_data.items():
        for sample in tqdm(data, desc=f"   Normalizing {letter}", leave=False):
            normalized = normalize_landmarks(sample)
            normalized_X.append(normalized)
            normalized_y.append(letter)
        print(f"   ✓ {letter}: {len(data)} samples normalized")
    
    custom_X = np.array(normalized_X, dtype=np.float32)
    custom_y = np.array(normalized_y)
    
    print(f"\n✓ Normalized {len(custom_X)} samples")
    
    # Step 5: Load original dataset
    print("\n" + "=" * 60)
    print("STEP 5: LOAD ORIGINAL DATASET")
    print("=" * 60)
    
    if not Path(INPUT_CSV).exists():
        print(f"✗ File not found: {INPUT_CSV}")
        return
    
    df_original = pd.read_csv(INPUT_CSV)
    print(f"✓ Loaded: {len(df_original):,} samples")
    
    # Step 6: Remove old letters
    print("\n" + "=" * 60)
    print("STEP 6: REMOVE OLD PROBLEM LETTERS")
    print("=" * 60)
    
    for letter in LETTERS_TO_REPLACE:
        count = (df_original['label'] == letter).sum()
        print(f"   Removing {letter}: {count:,} samples")
    
    mask = ~df_original['label'].isin(LETTERS_TO_REPLACE)
    df_filtered = df_original[mask]
    
    print(f"\n✓ After removal: {len(df_filtered):,} samples")
    
    # Step 7: Merge with new data
    print("\n" + "=" * 60)
    print("STEP 7: MERGE NEW DATA")
    print("=" * 60)
    
    custom_df = pd.DataFrame(
        np.column_stack([custom_y, custom_X]),
        columns=['label'] + [f'coord_{i}' for i in range(63)]
    )
    
    df_merged = pd.concat([df_filtered, custom_df], ignore_index=True)
    
    print(f"   Original (filtered): {len(df_filtered):,}")
    print(f"   Replaced letters (D,F,G,S,V): {sum(len(augmented_data[l]) for l in LETTERS_TO_REPLACE if l in augmented_data):,}")
    print(f"   New classes (Nonsense): {sum(len(augmented_data[c]) for c in CLASSES_TO_ADD if c in augmented_data):,}")
    print(f"   Total: {len(df_merged):,}")
    
    # Step 8: Shuffle and save
    print("\n" + "=" * 60)
    print("STEP 8: SHUFFLE AND SAVE")
    print("=" * 60)
    
    df_merged = df_merged.sample(frac=1, random_state=42).reset_index(drop=True)
    df_merged.to_csv(OUTPUT_CSV, index=False)
    
    print(f"✓ Saved: {OUTPUT_CSV}")
    
    # Final report
    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    
    print(f"\nTotal samples: {len(df_merged):,}")
    print(f"Classes: {df_merged['label'].nunique()}")
    
    print("\nPer-letter breakdown:")
    print("-" * 50)
    
    letter_counts = df_merged['label'].value_counts().sort_index()
    original_counts = df_original['label'].value_counts()
    
    for letter in sorted(letter_counts.index):
        count = letter_counts[letter]
        original_count = original_counts.get(letter, 0)
        
        if letter in LETTERS_TO_REPLACE:
            added = count - 0  # All samples are new
            print(f"  {letter}: {count:5,} (replaced {original_count:,} with {added:,} new)")
        elif letter in CLASSES_TO_ADD:
            print(f"  {letter}: {count:5,} (NEW class for gibberish rejection)")
        else:
            print(f"     {letter}: {count:5,} (unchanged)")
    
    print("\n" + "=" * 60)
    print("NEXT STEP")
    print("=" * 60)
    print("Update train.py:")
    print(f"  CSV_PATH = '{OUTPUT_CSV}'")
    print("\nThen retrain:")
    print("  python train.py")
    print("=" * 60)
    print("\n Pipeline complete!")

if __name__ == "__main__":
    main()
