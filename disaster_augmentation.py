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

# ============== CONFIGURATION ==============
BASE_DIR = r"C:\Users\Admin\deepfake"
fake_source = os.path.join(BASE_DIR, "train/fake")
fake_output = os.path.join(BASE_DIR, "train/fake_augmented")
os.makedirs(fake_output, exist_ok=True)

# How many augmented versions per image
AUGMENTATIONS_PER_IMAGE = 15

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
        return Image.open(buffer)
    
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
        
        # Random rectangular region
        x1, y1 = random.randint(0, w//2), random.randint(0, h//2)
        x2, y2 = random.randint(w//2, w), random.randint(h//2, h)
        
        # Blur that region
        region = img_array[y1:y2, x1:x2]
        blurred = cv2.GaussianBlur(region, (15, 15), 0)
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
            except Exception as e:
                print(f"[WARNING] Augmentation failed: {e}")
                continue
        
        return img


# ============== MAIN AUGMENTATION LOOP ==============

def augment_dataset():
    """Generate augmented versions of all fake images"""
    
    print(f"[START] Augmenting fake disaster images...")
    print(f"        Source: {fake_source}")
    print(f"        Output: {fake_output}")
    print(f"        Augmentations per image: {AUGMENTATIONS_PER_IMAGE}")
    
    # Get all fake images
    fake_images = [f for f in os.listdir(fake_source) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(fake_images) == 0:
        print("[ERROR] No images found in fake source directory!")
        return
    
    print(f"\n[INFO] Found {len(fake_images)} fake images")
    print(f"[INFO] Will generate {len(fake_images) * AUGMENTATIONS_PER_IMAGE} total images\n")
    
    augmenter = DisasterImageAugmenter()
    total_generated = 0
    
    for img_file in tqdm(fake_images, desc="Augmenting"):
        try:
            img_path = os.path.join(fake_source, img_file)
            img = Image.open(img_path).convert('RGB')
            
            # Save original
            base_name = os.path.splitext(img_file)[0]
            img.save(os.path.join(fake_output, f"{base_name}_original.jpg"))
            total_generated += 1
            
            # Generate augmented versions
            for i in range(AUGMENTATIONS_PER_IMAGE):
                # Apply 2-4 random augmentations
                num_augs = random.randint(2, 4)
                aug_img = augmenter.apply_random_augmentations(img.copy(), num_augs)
                
                # Save augmented image
                output_name = f"{base_name}_aug_{i:03d}.jpg"
                aug_img.save(os.path.join(fake_output, output_name), quality=95)
                total_generated += 1
                
        except Exception as e:
            print(f"\n[ERROR] Failed to process {img_file}: {e}")
            continue
    
    print(f"\n[DONE] Generated {total_generated} images")
    print(f"[INFO] Original images: {len(fake_images)}")
    print(f"[INFO] Augmented images: {total_generated - len(fake_images)}")
    print(f"\n[NEXT STEP] Update your training script:")
    print(f"            train_dirs = [")
    print(f"                (r'{BASE_DIR}\\train\\real', None, 0),")
    print(f"                (r'{fake_output}', None, 1)  # Use augmented fake images")
    print(f"            ]")


if __name__ == "__main__":
    augment_dataset()