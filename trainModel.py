import os
import random
import shutil
from ultralytics import YOLO

def main():
    # ==========================================
    # 1. CONFIGURATION & DIRECTORY SETUP
    # ==========================================
    RAW_DATA_DIR = r"C:\Users\Danver Dawn\Documents\cnnModelV2\pili_nut_dataset"
    SPLIT_DIR = "dataset_split"

    POSSIBLE_CLASSES = ["PILI 1", "PILI 2", "PILI 3"]
    CLASSES = [d for d in os.listdir(RAW_DATA_DIR) if os.path.isdir(os.path.join(RAW_DATA_DIR, d))] if os.path.exists(RAW_DATA_DIR) else POSSIBLE_CLASSES

    VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

    if os.path.exists(SPLIT_DIR):
        shutil.rmtree(SPLIT_DIR)

    for split in ["images/train", "images/val", "labels/train", "labels/val"]:
        os.makedirs(os.path.join(SPLIT_DIR, split), exist_ok=True)

    print(f"Checking dataset at: {RAW_DATA_DIR}")
    print(f"Found class folders: {CLASSES}")
    print("Starting dataset validation and split into 80% Training and 20% Validation...")

    total_images_copied = 0
    total_labels_copied = 0

    # ==========================================
    # 2. ROBUST DATASET SPLITTING LOGIC
    # ==========================================
    for class_folder in CLASSES:
        folder_path = os.path.join(RAW_DATA_DIR, class_folder)
        
        if not os.path.exists(folder_path):
            continue

        all_files = os.listdir(folder_path)
        images = [f for f in all_files if f.lower().endswith(VALID_EXTENSIONS)]
        
        if not images:
            print(f"⚠️ Warning: No valid images found in '{folder_path}'.")
            continue

        random.seed(42)
        random.shuffle(images)

        split_idx = int(len(images) * 0.8)
        train_imgs = set(images[:split_idx])

        for img_name in images:
            base_name = os.path.splitext(img_name)[0]
            txt_name = base_name + ".txt"
            
            img_src = os.path.join(folder_path, img_name)
            txt_src = os.path.join(folder_path, txt_name)

            target_subfolder = "train" if img_name in train_imgs else "val"

            # Using os.link for instant setup on 10 GB dataset
            try:
                os.link(img_src, os.path.join(SPLIT_DIR, "images", target_subfolder, img_name))
            except Exception:
                shutil.copy(img_src, os.path.join(SPLIT_DIR, "images", target_subfolder, img_name))
                
            total_images_copied += 1

            if os.path.exists(txt_src):
                try:
                    os.link(txt_src, os.path.join(SPLIT_DIR, "labels", target_subfolder, txt_name))
                except Exception:
                    shutil.copy(txt_src, os.path.join(SPLIT_DIR, "labels", target_subfolder, txt_name))
                total_labels_copied += 1

    print(f"✅ Split Complete! Processed {total_images_copied} images and {total_labels_copied} label files.")

    if total_images_copied == 0:
        raise FileNotFoundError(f"CRITICAL ERROR: No images found inside '{RAW_DATA_DIR}'. Check folder contents.")

    # ==========================================
    # 3. CREATE DATA.YAML
    # ==========================================
    yaml_content = f"""
path: {os.path.abspath(SPLIT_DIR)} 
train: images/train
val: images/val

names:
  0: PILI 1
  1: PILI 2
  2: PILI 3
"""

    with open("data.yaml", "w") as f:
        f.write(yaml_content.strip())

    print("✅ 'data.yaml' generated successfully.")

    # ==========================================
    # 4. YOLOV8 TRAINING & EXPORT
    # ==========================================
    print("\n🚀 Starting YOLOv8 training...")

    model = YOLO("yolov8n.pt")

    model.train(
        data="data.yaml",
        epochs=100,
        patience=20,
        imgsz=640,
        batch=16,
        
        optimizer='AdamW',   # Superior texture and small-feature convergence
        box=8.5,             # Penalizes wide/loose bounding boxes
        cls=1.5,             # Strongly penalizes wrong classification
        dfl=2.0,             # Enhances fine boundary details
        
        hsv_h=0.6,           # Drastically shifts hue (forces model to ignore specific background colors)
        hsv_s=0.8,           # Simulates varying conveyor belt reflections
        hsv_v=0.6,           # Simulates harsh shadows and changing overhead lights
        
        degrees=180.0,       # Full 180-degree rotation
        fliplr=0.5,          # 50% horizontal flip
        flipud=0.5,          # 50% vertical flip
        scale=0.5,           # Scales object sizes +/- 50%
        
        mosaic=1.0,          # Stitches 4 images into 1 frame
        mixup=0.15,          # Overlays two images to break background dependency
        workers=2            # Limits parallel data loading threads to prevent memory overload
    )

    print("\n⚙️ Exporting trained model to NCNN format...")
    model.export(format="ncnn", imgsz=640)
    print("\n🎉 Training and NCNN export complete!")

# CRITICAL ENTRY POINT PROTECTOR FOR WINDOWS
if __name__ == '__main__':
    main()