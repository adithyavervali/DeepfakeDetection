# Deepfake Disaster Image Detection

Advanced image augmentation pipeline for training deepfake disaster image detection models.

## Features

- Specialized augmentation techniques for fake disaster images
- Multiple augmentation methods including:
  - Noise addition
  - JPEG compression artifacts
  - Color shifts
  - Local blur effects
  - Edge enhancement
  - Chromatic aberration
  - Perspective warping
  - Weather effects

## Setup

1. Create directory structure:
   ```
   deepfake/
   ├── train/
   │   ├── fake/         # Original fake images
   │   ├── fake_augmented/   # Augmented output
   │   └── real/         # Real disaster images
   ```

2. Place your training images in the appropriate directories

3. Run the augmentation script:
   ```python
   python disaster_augmentation.py
   ```

## Configuration

Edit the following variables in `disaster_augmentation.py`:
- `BASE_DIR`: Base directory path
- `AUGMENTATIONS_PER_IMAGE`: Number of augmented versions to generate per image

## Requirements

- Python 3.7+
- OpenCV
- PIL
- NumPy
- tqdm