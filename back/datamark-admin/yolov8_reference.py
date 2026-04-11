import os
from pathlib import Path
import shutil
import cv2
import pandas as pd
import matplotlib.pyplot as plt

from ultralytics import YOLO


# ===========================
# 配置区
# ===========================

DATA_YAML = "/autodl-tmp/dataset_yolo/data.yaml"

DATA_ROOT = Path("/autodl-tmp/dataset_yolo")

MODEL_NAME = "yolov8s.pt"

PROJECT_DIR = "./runs_power_inspection"
EXP_NAME = "exp_v1"

EPOCHS = 120
IMG_SIZE = 640
BATCH = 16
DEVICE = 0

CONF_THRESH = 0.25

# 可视化输出目录
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
        save=True,
        save_period=5,
        plots=True
    )

    return results


# ===========================
# 2️⃣ 评估（val + test）
# ===========================

def evaluate(model_path):

    model = YOLO(model_path)

    val_metrics = model.val(
        data=DATA_YAML,
        split="val",
        conf=CONF_THRESH
    )

    test_metrics = model.val(
        data=DATA_YAML,
        split="test",
        conf=CONF_THRESH
    )

    return val_metrics, test_metrics


# ===========================
# 3️⃣ 导出指标 CSV
# ===========================

def export_metrics(val_metrics, test_metrics):

    rows = []

    for name, m in zip(
        ["val", "test"],
        [val_metrics, test_metrics]
    ):
        rows.append({
            "split": name,
            "mAP50": m.box.map50,
            "mAP50-95": m.box.map,
            "precision": m.box.mp,
            "recall": m.box.mr
        })

    df = pd.DataFrame(rows)
    df.to_csv("metrics_summary.csv", index=False)

    print("📊 指标已保存 metrics_summary.csv")


# ===========================
# 4️⃣ 可视化 GT 标注
# ===========================

def visualize_gt(split="val", num_images=20):
    img_dir = DATA_ROOT / "images" / split
    label_dir = DATA_ROOT / "labels" / split
    imgs = list(img_dir.glob("*.jpg"))[:num_images]

    for img_path in imgs:

        img = cv2.imread(str(img_path))
        h, w, _ = img.shape

        label_path = label_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
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
            cv2.putText(
                img,
                str(int(cid)),
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        cv2.imwrite(str(VIS_GT_DIR / img_path.name), img)

    print("🟢 GT 可视化完成:", VIS_GT_DIR)


# ===========================
# 5️⃣ 推理 + 可视化预测
# ===========================

def visualize_predictions(model_path, split="test", num_images=20):

    model = YOLO(model_path)

    img_dir = DATA_ROOT / "images" / split
    imgs = list(img_dir.glob("*.jpg"))[:num_images]

    for img_path in imgs:

        results = model.predict(
            source=str(img_path),
            conf=CONF_THRESH,
            save=False
        )

        im = results[0].plot()

        cv2.imwrite(
            str(VIS_PRED_DIR / img_path.name),
            im
        )

    print("🔵 预测可视化完成:", VIS_PRED_DIR)


# ===========================
# 主流程
# ===========================

def main():

    train()

    best_list = list(Path(PROJECT_DIR).rglob("best.pt"))

    assert len(best_list) > 0, "❌ 没有找到 best.pt"

    best_model = max(best_list, key=lambda x: x.stat().st_mtime)

    print("✅ 使用模型:", best_model)

    val_metrics, test_metrics = evaluate(best_model)

    export_metrics(val_metrics, test_metrics)

    visualize_gt("val", 30)

    visualize_predictions(best_model, "test", 30)


if __name__ == "__main__":
    main()