import os
import shutil
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from tqdm import tqdm
import hashlib

# paths
preprocessed_dir = " "
label_dir = " "
aug_img_dir = " "
aug_lbl_dir = " "


if os.path.exists(aug_img_dir):
    shutil.rmtree(aug_img_dir)
if os.path.exists(aug_lbl_dir):
    shutil.rmtree(aug_lbl_dir)
os.makedirs(aug_img_dir, exist_ok=True)
os.makedirs(aug_lbl_dir, exist_ok=True)


def rotate(img, angle): return img.rotate(angle)
def adjust_saturation(img, factor): return ImageEnhance.Color(img).enhance(factor)
def add_salt_pepper_noise(img_np, amount=0.004):
    out = np.copy(img_np)
    row, col, ch = out.shape
    num_salt = np.ceil(amount * out.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in out.shape[:2]]
    out[coords[0], coords[1], :] = 255
    num_pepper = np.ceil(amount * out.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in out.shape[:2]]
    out[coords[0], coords[1], :] = 0
    return out
def apply_blur(img_np): return cv2.GaussianBlur(img_np, (5, 5), sigmaX=2)


def safe_filename(name, suffix):
    hash_name = hashlib.md5(name.encode()).hexdigest()[:12]
    return f"{hash_name}_{suffix}"


original_files = [f for f in os.listdir(preprocessed_dir) if f.endswith(".jpg")]
total_required = 6696
copies_per_image = total_required // len(original_files)
remainder = total_required % len(original_files)

print(f"\n Generating {copies_per_image} copies per image, +1 for {remainder} images.")

count = 0
for idx, img_file in enumerate(tqdm(original_files)):
    base_name = os.path.splitext(img_file)[0]
    img_path = os.path.join(preprocessed_dir, img_file)
    label_path = os.path.join(label_dir, base_name + ".txt")

    try:
        img = Image.open(img_path)

  
        repeat = copies_per_image + (1 if idx < remainder else 0)

        for i in range(repeat):
            aug = rotate(img, 90 if i % 2 == 0 else -45)
            aug = adjust_saturation(aug, 1.0 + (0.1 if i % 2 == 0 else -0.1))
            aug_np = np.array(aug)
            aug_np = add_salt_pepper_noise(aug_np)
            aug_np = apply_blur(aug_np)
            aug = Image.fromarray(aug_np)

            unique_name = safe_filename(base_name + str(i), "aug")
            aug.save(os.path.join(aug_img_dir, unique_name + ".jpg"))
            if os.path.exists(label_path):
                shutil.copy(label_path, os.path.join(aug_lbl_dir, unique_name + ".txt"))

            count += 1

    except Exception as e:
        print(f" Error processing {img_file}: {e}")

print(f"\n Augmentation completed. Total generated images: {count}")
