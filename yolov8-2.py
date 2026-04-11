import os
from pathlib import Path
import cv2
import pandas as pd

from ultralytics import YOLO


# ===========================
# 配置区
# ===========================

DATA_YAML = "autodl-tmp/dataset_yolo/data.yaml"

# ✅ 修正路径（关键！！！）
DATA_ROOT = Path("/root/autodl-tmp/dataset_yolo")

MODEL_NAME = "yolov8s.pt"
PROJECT_DIR = "./models"
EXP_NAME = "exp_v1"

EPOCHS = 120
IMG_SIZE = 640
BATCH = 16
DEVICE = 0

CONF_THRESH = 0.25

# 输出目录
VIS_GT_DIR = Path("visualizations/gt")
VIS_PRED_DIR = Path("visualizations/pred")

VIS_GT_DIR.mkdir(parents=True, exist_ok=True)
VIS_PRED_DIR.mkdir(parents=True, exist_ok=True)


# ===========================
# 1️⃣ 训练
# ===========================

def train():
    model = YOLO(MODEL_NAME)

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,
        project=PROJECT_DIR,
        name=EXP_NAME,
        exist_ok=True,
        save=True,
        save_period=5,
        plots=True
    )

    return results


# ===========================
# 2️⃣ 评估
# ===========================

def evaluate(model_path):

    model = YOLO(model_path)

    val_metrics = model.val(data=DATA_YAML, split="val", conf=CONF_THRESH)
    test_metrics = model.val(data=DATA_YAML, split="test", conf=CONF_THRESH)

    return val_metrics, test_metrics


# ===========================
# 3️⃣ 可视化 GT（重点修复）
# ===========================

def visualize_gt(split="val", num_images=20):

    img_dir = DATA_ROOT / "images" / split
    label_dir = DATA_ROOT / "labels" / split

    imgs = list(img_dir.glob("*.jpg"))

    print(f"📂 {split} 图片数量:", len(imgs))

    imgs = imgs[:num_images]

    for img_path in imgs:

        img = cv2.imread(str(img_path))
        if img is None:
            print("❌ 读图失败:", img_path)
            continue

        h, w, _ = img.shape

        label_path = label_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            print("⚠️ 没标注:", img_path.name)
            continue

        with open(label_path) as f:
            lines = f.readlines()

        for line in lines:
            cid, xc, yc, bw, bh = map(float, line.split())

            xc *= w
            yc *= h
            bw *= w
            bh *= h

            x1 = int(xc - bw / 2)
            y1 = int(yc - bh / 2)
            x2 = int(xc + bw / 2)
            y2 = int(yc + bh / 2)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imwrite(str(VIS_GT_DIR / img_path.name), img)

    print("🟢 GT 可视化完成:", VIS_GT_DIR)


# ===========================
# 4️⃣ 预测可视化
# ===========================

def visualize_predictions(model_path, split="test", num_images=20):

    model = YOLO(model_path)

    img_dir = DATA_ROOT / "images" / split
    imgs = list(img_dir.glob("*.jpg"))

    print(f"📂 {split} 图片数量:", len(imgs))

    imgs = imgs[:num_images]

    for img_path in imgs:

        results = model.predict(
            source=str(img_path),
            conf=CONF_THRESH,
            save=False
        )

        im = results[0].plot()

        cv2.imwrite(str(VIS_PRED_DIR / img_path.name), im)

    print("🔵 预测可视化完成:", VIS_PRED_DIR)


# ===========================
# 主流程
# ===========================

def main():

    # 训练
    results = train()

    # ⭐直接拿训练输出路径（关键）
    best_model = Path(results.save_dir) / "weights/best.pt"

    assert best_model.exists(), f"❌ 没找到: {best_model}"

    print("✅ 使用模型:", best_model)

    evaluate(best_model)

    # ✅ 这里明确：
    visualize_gt("test", 30)        # 测试集 GT
    visualize_predictions(best_model, "test", 30)  # 测试集预测


if __name__ == "__main__":
    main()