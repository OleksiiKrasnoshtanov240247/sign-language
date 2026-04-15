# NGT Fingerspelling Recognition Dataset

## Overview

This dataset contains hand landmark data for Dutch Sign Language (Nederlandse Gebarentaal - NGT) fingerspelling recognition. The dataset supports recognition of all 26 letters of the alphabet, split into static poses and dynamic gestures.

## Dataset Statistics

### Static Letters (21 letters)

- **Letters**: A, B, C, D, E, F, G, I, K, L, M, N, O, P, Q, R, S, T, V, W, Y
- **Total Samples**: 42,909
- **Features per Sample**: 42 (21 hand landmarks × 2 coordinates)
- **File**: `ngt_static_keypoint.csv`
- **Model Accuracy**: 99.5%

### Dynamic Letters (5 letters)

- **Letters**: H, J, U, X, Z
- **Total Samples**: 16,128
- **Features per Sample**: 84 (42 current frame + 42 delta features)
- **File**: `ngt_dynamic_keypoint.csv`
- **Model Accuracy**: 100%

## Data Collection

### Hardware & Software

- **Camera**: Standard webcam at 960×540 resolution
- **Hand Tracking**: MediaPipe Hands (v0.10+)
- **Frame Rate**: ~30 fps
- **Detection Settings**: 
  - `min_detection_confidence`: 0.7
  - `min_tracking_confidence`: 0.5

### Collection Scripts

- **Static Letters**: `../notebooks/collection.ipynb`
- **Dynamic Letters**: `../notebooks/dynamic_collection.ipynb`

### Collection Process

#### Static Letters

Static letters represent hand poses that don't require movement. Data was collected by:

1. User selects a letter key to begin collection
2. User forms the hand pose for that letter
3. System continuously captures normalized hand landmarks at ~30 fps
4. User presses SPACE to pause/resume collection
5. Target: ~2000 samples per letter

**Workflow**:

- Press letter key to activate collection
- Press SPACE to pause between attempts
- System shows real-time progress and remaining letters
- Hand skeleton overlay provides visual feedback

#### Dynamic Letters

Dynamic letters (H, J, U, X, Z) require movement for proper recognition. These letters were collected using a rolling window approach:

1. User selects a dynamic letter key
2. System maintains a 5-frame rolling buffer
3. User performs the gesture repeatedly
4. System captures overlapping 5-frame sequences
5. Each sample includes current pose + motion delta

**Key Differences**:

- Rolling window size: 5 frames (~167ms at 30fps)
- Dense sampling: ~30 samples per second during gesture
- Motion features computed from frame deltas
- Higher sample target: 2000+ per letter

**Why This Approach?**

The rolling window with overlapping samples provides:

- Dense coverage of gesture phases
- Temporal information through delta features
- Robustness to timing variations
- Multiple examples from each gesture performance

## Feature Engineering

### Static Features (42 values)

Each sample contains normalized 2D coordinates for 21 hand landmarks:

```
[x0, y0, x1, y1, ..., x20, y20]
```

**Normalization Process**:

1. Extract (x, y) pixel coordinates from MediaPipe
2. Translate relative to wrist (landmark 0): `x' = x - wrist_x`
3. Flatten to 1D array
4. Scale by max absolute value: `normalized = coords / max(|coords|)`

**Result**: All coordinates in range [-1, 1], scale-invariant, translation-invariant

### Dynamic Features (84 values)

Each sample contains current frame landmarks plus motion deltas:

```
[current_42_features, delta_42_features]
```

**Delta Computation**:

1. Maintain rolling buffer of last 5 normalized frames
2. Current frame: `frame[t]` (42 features)
3. Historical frame: `frame[t-5]` (42 features)
4. Delta: `delta = frame[t] - frame[t-5]`
5. Concatenate: `[frame[t], delta]` (84 features)

**Why Deltas Work**:

- Captures movement direction and magnitude
- Normalized deltas are scale-invariant
- Distinguishes H from U (identical poses, different motion)
- Robust to camera distance variations

## File Formats

### CSV Structure

**Static CSV** (`ngt_static_keypoint.csv`):

```
label_index,x0,y0,x1,y1,...,x20,y20
0,-0.1,0.2,0.3,-0.4,...,0.5,-0.1
```

**Dynamic CSV** (`ngt_dynamic_keypoint.csv`):

```
label_index,x0,y0,...,x20,y20,dx0,dy0,...,dx20,dy20
7,0.0,0.0,...,0.4,-0.2,-0.01,0.02,...,0.03,-0.01
```

### Label Mapping

Labels use alphabetical indices (A=0, B=1, ..., Z=25):

```python
alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
letter = alphabet[label_index]
```

## Models

### Static Model

- **Type**: Random Forest Classifier
- **Features**: 42 normalized landmark coordinates
- **Training**: 80/20 train/test split, stratified
- **Hyperparameters**: 100 estimators, default sklearn settings
- **Performance**: 99.5% accuracy on test set
- **File**: `../models/ngt_static_classifier.pkl`

### Dynamic Model

- **Type**: Random Forest Classifier
- **Features**: 84 (42 current + 42 delta)
- **Training**: 80/20 train/test split, stratified
- **Hyperparameters**: 100 estimators, default sklearn settings
- **Performance**: 100% accuracy on test set
- **File**: `../models/ngt_dynamic_classifier.pkl`

### Inference Strategy

A supervisor module routes predictions based on wrist motion:

1. Track wrist position over 10-frame window
2. Calculate cumulative motion distance
3. If `motion > threshold`: use dynamic model
4. Else: use static model

**Motion Threshold**: 0.1 (normalized coordinate space)

This hybrid approach ensures:
- Fast recognition of static letters
- Accurate capture of dynamic gestures
- Minimal false triggering between models

## Usage Example

```python
import joblib
import numpy as np
from collections import deque

# Load models
static_model = joblib.load('../models/ngt_static_classifier.pkl')
dynamic_model = joblib.load('../models/ngt_dynamic_classifier.pkl')

# For static prediction
normalized_landmarks = [...]  # 42 features
prediction = static_model.predict([normalized_landmarks])[0]
letter = chr(prediction + ord('A'))

# For dynamic prediction
landmark_buffer = deque(maxlen=5)
# ... fill buffer with 5 frames ...
current = landmark_buffer[-1]
old = landmark_buffer[0]
delta = [c - o for c, o in zip(current, old)]
features = current + delta  # 84 features
prediction = dynamic_model.predict([features])[0]
letter = chr(prediction + ord('A'))
```

## Data Quality Notes

### Strengths

- High sample counts per letter (2000-4000)
- Consistent normalization pipeline
- Real-world variation in hand positions and distances
- Dense temporal sampling for dynamic gestures

### Limitations

- Single collector (limited hand size/style variation)
- Controlled lighting conditions
- Limited to frontal camera angle
- Mirror-flipped during collection (camera flip applied)

### Known Issues

- Some static letters show rotational variation due to normalization
- H and U have identical static poses (correctly separated by motion)
- Model trained on M2 Mac, cross-platform compatibility verified

## Dataset Provenance

This dataset combines:
- **ASL Foundation**: Cleaned ASL fingerspelling dataset for letters not in NGT collection
- **NGT Samples**: Newly collected data for Dutch-specific letter formations
- **Filtering**: Dynamic letters (H, J, U, X, Z) removed from static dataset

The hybrid approach leverages existing quality data while ensuring NGT-specific accuracy for letters that differ from ASL.

## Citation

If you use this dataset, please reference:
- MediaPipe Hands: https://google.github.io/mediapipe/solutions/hands.html
- Collection methodology: Rolling window with delta features for dynamic gesture recognition

## Version History

- **v1.0** (2025-01): Initial release
  - 42,909 static samples (21 letters)
  - 16,128 dynamic samples (5 letters)
  - Random Forest classifiers
  - 99.5% static accuracy, 100% dynamic accuracy

## License

Dataset and models are provided for research and educational purposes.

## Contact

For questions about data collection methodology or model architecture, please refer to the collection notebooks in `../notebooks/`.