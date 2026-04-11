"""
数据集分层抽样脚本

将"每类一个文件夹（图像+XML标注）"的原始目录分层划分为 train/val/test。

用法示例：
    python split.py --raw_root /data/raw --out_root /data/dataset_split \
                    --train_ratio 0.7 --val_ratio 0.2 --test_ratio 0.1

输入目录结构：
    raw_root/
      class_a/*.jpg + *.xml
      class_b/*.jpg + *.xml
      ...

输出目录结构：
    out_root/
      train/images/*.jpg
      train/annotations/*.xml
      val/images/*.jpg
      val/annotations/*.xml
      test/images/*.jpg
      test/annotations/*.xml
"""

import argparse
import os
import random
import shutil
from pathlib import Path

# ==============================
# 默认配置（可通过命令行覆盖）
# ==============================

DEFAULT_RAW_ROOT = "F:/DOWN/sample-1000 -1"
DEFAULT_OUT_ROOT = "./dataset_split"

DEFAULT_TRAIN_RATIO = 0.7
DEFAULT_VAL_RATIO = 0.2
DEFAULT_TEST_RATIO = 0.1

IMG_EXTS = [".jpg", ".png", ".jpeg"]


# ==============================
# UTILS
# ==============================

def mkdir(p):
    os.makedirs(p, exist_ok=True)


def copy_pair(img_path, xml_path, img_dst, xml_dst):
    shutil.copy(img_path, img_dst)
    shutil.copy(xml_path, xml_dst)


def parse_args():
    parser = argparse.ArgumentParser(description='数据集分层抽样划分工具')
    parser.add_argument('--raw_root', type=str, default=DEFAULT_RAW_ROOT,
                        help='原始数据根目录（每个类别一个子文件夹，每个子文件夹包含图像和XML）')
    parser.add_argument('--out_root', type=str, default=DEFAULT_OUT_ROOT,
                        help='输出目录')
    parser.add_argument('--train_ratio', type=float, default=DEFAULT_TRAIN_RATIO,
                        help='训练集比例 (默认 0.7)')
    parser.add_argument('--val_ratio', type=float, default=DEFAULT_VAL_RATIO,
                        help='验证集比例 (默认 0.2)')
    parser.add_argument('--test_ratio', type=float, default=DEFAULT_TEST_RATIO,
                        help='测试集比例 (默认 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认 42)')
    return parser.parse_args()


# ==============================
# MAIN
# ==============================

def main():
    args = parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    train_ratio = args.train_ratio
    val_ratio = args.val_ratio

    if not raw_root.exists():
        raise FileNotFoundError(f"原始数据目录不存在: {raw_root}")

    random.seed(args.seed)

    splits = ["train", "val", "test"]

    for s in splits:
        mkdir(out_root / s / "images")
        mkdir(out_root / s / "annotations")

    # 遍历每个类别文件夹
    for cls in sorted(os.listdir(raw_root)):

        cls_dir = raw_root / cls
        if not cls_dir.is_dir():
            continue

        print(f"\n📂 Processing class: {cls}")

        images = []

        for f in os.listdir(cls_dir):
            if Path(f).suffix.lower() in IMG_EXTS:
                images.append(cls_dir / f)

        images.sort()
        random.shuffle(images)

        total = len(images)

        n_train = int(total * train_ratio)
        n_val = int(total * val_ratio)

        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]

        print(f"  total={total}, train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}")

        def process(split_name, img_list):
            for img_path in img_list:
                xml_path = img_path.with_suffix(".xml")
                if not xml_path.exists():
                    print(f"⚠ Missing XML: {xml_path}")
                    continue
                dst_img = out_root / split_name / "images" / img_path.name
                dst_xml = out_root / split_name / "annotations" / xml_path.name
                copy_pair(img_path, xml_path, dst_img, dst_xml)

        process("train", train_imgs)
        process("val", val_imgs)
        process("test", test_imgs)

    print("\n✅ Stratified split finished!")
    print(f"   输出目录: {out_root}")


if __name__ == "__main__":
    main()
