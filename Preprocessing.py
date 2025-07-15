import os
from PIL import Image
from tqdm import tqdm

input_image_dir = " "
preprocessed_image_dir = " "
target_size = (640, 640)

os.makedirs(preprocessed_image_dir, exist_ok=True)

def preprocess_image(img_path, save_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize(target_size, Image.LANCZOS)
    img.save(save_path, format="JPEG")

for img_file in tqdm(os.listdir(input_image_dir)):
    if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
        in_path = os.path.join(input_image_dir, img_file)
        base = os.path.splitext(img_file)[0]
        out_path = os.path.join(preprocessed_image_dir, base + ".jpg")
        preprocess_image(in_path, out_path)

print(" Preprocessing done: Images resized and standardized.")
