"""
ADVANCED AUGMENTATION FOR FAKE DISASTER IMAGES
Run this script BEFORE training to generate more fake samples
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import random
from tqdm import tqdm
import argparse
import time

# ============== CONFIGURATION ==============
BASE_DIR = r"C:\Users\Admin\deepfake"
fake_source = os.path.join(BASE_DIR, "valid/fake")
fake_output = os.path.join(BASE_DIR, "valid/fake_augmented")
os.makedirs(fake_output, exist_ok=True)

# How many augmented versions per image (default, can be overridden via CLI)
AUGMENTATIONS_PER_IMAGE = 15
# Default multiplier (produce roughly multiplier * existing_count images)
DEFAULT_MULTIPLIER = 3

# ============== ADVANCED AUGMENTATION TECHNIQUES ==============
class DisasterImageAugmenter:
    """
    Specialized augmentation for fake disaster image detection
    Focuses on artifacts that AI-generated disaster images may have
    """
    
    @staticmethod
    def add_noise(img, intensity=0.05):
        """Add Gaussian noise (common in AI-generated images)"""
        img_array = np.array(img).astype(np.float32) / 255.0
        noise = np.random.normal(0, intensity, img_array.shape)
        noisy = np.clip(img_array + noise, 0, 1)
        return Image.fromarray((noisy * 255).astype(np.uint8))
    
    @staticmethod
    def jpeg_compression(img, quality=None):
        """Simulate JPEG compression artifacts"""
        if quality is None:
            quality = random.randint(60, 95)
        from io import BytesIO
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        # return a copied Image (load buffer while open) and ensure RGB
        out = Image.open(buffer).convert('RGB').copy()
        buffer.close()
        return out

    @staticmethod
    def color_shift(img):
        """Shift color channels (common in fake images)"""
        img_array = np.array(img)
        shifts = [random.randint(-20, 20) for _ in range(3)]
        
        for i in range(3):
            img_array[:, :, i] = np.clip(img_array[:, :, i].astype(np.int16) + shifts[i], 0, 255)
        
        return Image.fromarray(img_array.astype(np.uint8))
    
    @staticmethod
    def 局部模糊(img):
        """Local blur (simulates AI generation artifacts)"""
        img_array = np.array(img)
        h, w = img_array.shape[:2]

        # Random rectangular region (ensure width/height >= 10)
        x1 = random.randint(0, max(0, w - 10))
        y1 = random.randint(0, max(0, h - 10))
        x2 = random.randint(min(w, x1 + 10), w)
        y2 = random.randint(min(h, y1 + 10), h)

        # Blur that region only if valid
        if x2 > x1 and y2 > y1:
            region = img_array[y1:y2, x1:x2]
            # kernel size should be odd and <= region dims
            k = min(31, max(3, (min(region.shape[0], region.shape[1]) // 10) * 2 + 1))
            blurred = cv2.GaussianBlur(region, (k, k), 0)
            img_array[y1:y2, x1:x2] = blurred

        return Image.fromarray(img_array)
    
    @staticmethod
    def edge_enhancement(img):
        """Enhance edges (fake images often have weird edges)"""
        enhancer = ImageEnhance.Sharpness(img)
        factor = random.uniform(1.5, 3.0)
        return enhancer.enhance(factor)
    
    @staticmethod
    def chromatic_aberration(img):
        """Simulate chromatic aberration"""
        img_array = np.array(img)
        h, w = img_array.shape[:2]
        
        # Slight shift in color channels
        shift = random.randint(1, 3)
        r = img_array[:, :, 0]
        g = img_array[:, :, 1]
        b = img_array[:, :, 2]
        
        # Shift channels
        r = np.roll(r, shift, axis=1)
        b = np.roll(b, -shift, axis=1)
        
        img_array[:, :, 0] = r
        img_array[:, :, 2] = b
        
        return Image.fromarray(img_array)
    
    @staticmethod
    def contrast_adjustment(img):
        """Random contrast adjustment"""
        enhancer = ImageEnhance.Contrast(img)
        factor = random.uniform(0.7, 1.5)
        return enhancer.enhance(factor)
    
    @staticmethod
    def saturation_adjustment(img):
        """Random saturation adjustment"""
        enhancer = ImageEnhance.Color(img)
        factor = random.uniform(0.6, 1.8)
        return enhancer.enhance(factor)
    
    @staticmethod
    def rotation_small(img):
        """Small rotation"""
        angle = random.uniform(-15, 15)
        return img.rotate(angle, fillcolor=(128, 128, 128), expand=False)
    
    @staticmethod
    def perspective_warp(img):
        """Perspective transformation"""
        img_array = np.array(img)
        h, w = img_array.shape[:2]
        
        # Random perspective points
        margin = int(0.1 * min(w, h))
        src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst_points = np.float32([
            [random.randint(0, margin), random.randint(0, margin)],
            [w - random.randint(0, margin), random.randint(0, margin)],
            [w - random.randint(0, margin), h - random.randint(0, margin)],
            [random.randint(0, margin), h - random.randint(0, margin)]
        ])
        
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        warped = cv2.warpPerspective(img_array, matrix, (w, h))
        return Image.fromarray(warped)
    
    @staticmethod
    def add_weather_effect(img):
        """Simulate weather effects (rain, fog, etc.)"""
        img_array = np.array(img).astype(np.float32)
        
        effect_type = random.choice(['fog', 'brightness'])
        
        if effect_type == 'fog':
            # Add fog/haze
            fog_intensity = random.uniform(0.2, 0.5)
            fog = np.ones_like(img_array) * 200
            img_array = img_array * (1 - fog_intensity) + fog * fog_intensity
        
        elif effect_type == 'brightness':
            # Adjust brightness
            brightness = random.uniform(0.8, 1.3)
            img_array = img_array * brightness
        
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)

    @staticmethod
    def apply_random_augmentations(img, num_augs=3):
        """Apply multiple random augmentations"""
        augmentations = [
            DisasterImageAugmenter.add_noise,
            DisasterImageAugmenter.jpeg_compression,
            DisasterImageAugmenter.color_shift,
            DisasterImageAugmenter.局部模糊,
            DisasterImageAugmenter.edge_enhancement,
            DisasterImageAugmenter.chromatic_aberration,
            DisasterImageAugmenter.contrast_adjustment,
            DisasterImageAugmenter.saturation_adjustment,
            DisasterImageAugmenter.rotation_small,
            DisasterImageAugmenter.perspective_warp,
            DisasterImageAugmenter.add_weather_effect
        ]

        selected_augs = random.sample(augmentations, min(num_augs, len(augmentations)))

        for aug in selected_augs:
            try:
                img = aug(img)
                # ensure returned image is PIL RGB
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(np.array(img)).convert('RGB')
                else:
                    img = img.convert('RGB')
            except Exception as e:
                print(f"[WARNING] Augmentation failed: {e}")
                continue

        return img

# ============== MAIN AUGMENTATION LOOP ==============
def augment_dataset(source=None, output=None, per_image=None, multiplier=None, target_total=None, seed=None):
    """Generate augmented versions of fake images.
    - source: source folder with original images
    - output: where to save augmented images
    - per_image: augmentations per original image
    - multiplier: generate about multiplier * N images total (including originals)
    - target_total: explicit target total images in output (including originals)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    src = source or fake_source
    out = output or fake_output
    per_image = per_image if per_image is not None else AUGMENTATIONS_PER_IMAGE

    if not os.path.isdir(src):
        print(f"[ERROR] Source directory does not exist: {src}")
        return

    os.makedirs(out, exist_ok=True)

    # collect originals from source
    fake_images = [f for f in os.listdir(src)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if len(fake_images) == 0:
        print("[ERROR] No images found in source directory!")
        return

    orig_count = len(fake_images)
    print(f"[START] Found {orig_count} source images in {src}")

    # Determine required number of augmented images
    if target_total is not None:
        required_total = int(target_total)
    elif multiplier is not None:
        required_total = int(orig_count * multiplier)
    else:
        required_total = orig_count * (per_image + 1)  # original + per_image each

    # Count current images in output to avoid duplicating work
    existing_out_images = [f for f in os.listdir(out)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    current_out_count = len(existing_out_images)
    print(f"[INFO] Output currently has {current_out_count} images, target is {required_total}")

    # If output already meets or exceeds required, skip generation
    if current_out_count >= required_total:
        print("[INFO] Output already meets or exceeds target. No augmentation performed.")
        return

    to_generate = required_total - current_out_count
    print(f"[INFO] Need to generate approximately {to_generate} images")

    augmenter = DisasterImageAugmenter()
    generated = 0
    start_time = time.time()

    # iterate through source images repeatedly until we hit target
    idx = 0
    while generated < to_generate:
        img_file = fake_images[idx % orig_count]
        try:
            img_path = os.path.join(src, img_file)
            img = Image.open(img_path).convert('RGB')

            base_name = os.path.splitext(img_file)[0]
            # Optionally save the original into output once (if not present)
            orig_out_name = f"{base_name}_original.jpg"
            if orig_out_name not in existing_out_images and generated < to_generate:
                img.save(os.path.join(out, orig_out_name), quality=95)
                existing_out_images.append(orig_out_name)
                generated += 1

            # number of augmentations to create from this source this pass
            # attempt to evenly distribute work across files
            create_count = min(per_image, to_generate - generated)

            for i in range(create_count):
                num_augs = random.randint(2, 4)
                aug_img = augmenter.apply_random_augmentations(img.copy(), num_augs)
                # unique name with timestamp and counter to avoid collisions
                ts = int(time.time() * 1000) % 1000000
                output_name = f"{base_name}_aug_{idx}_{i}_{ts}.jpg"
                aug_img.save(os.path.join(out, output_name), quality=95)
                generated += 1
                if generated >= to_generate:
                    break

        except Exception as e:
            print(f"[ERROR] Failed to process {img_file}: {e}")

        idx += 1

    elapsed = time.time() - start_time
    print(f"[DONE] Generated {generated} new images in {elapsed:.1f}s")
    total_now = len([f for f in os.listdir(out) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"[INFO] Output now contains {total_now} images (target was {required_total})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment fake images to boost dataset size")
    parser.add_argument("--source", "-s", help="Source folder (default valid/fake)", default=None)
    parser.add_argument("--output", "-o", help="Output folder (default valid/fake_augmented)", default=None)
    parser.add_argument("--per-image", "-p", type=int, help="Augmentations per source image", default=None)
    parser.add_argument("--multiplier", "-m", type=float, help="Target multiplier (eg 3 to make 3x total)", default=None)
    parser.add_argument("--target-total", "-t", type=int, help="Exact target total images in output (including originals)", default=None)
    parser.add_argument("--seed", type=int, help="Random seed", default=None)
    args = parser.parse_args()

    augment_dataset(source=args.source, output=args.output, per_image=args.per_image,
                    multiplier=args.multiplier, target_total=args.target_total, seed=args.seed)