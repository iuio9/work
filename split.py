import os
import random
import shutil
from pathlib import Path

# ==============================
# CONFIG
# ==============================

RAW_ROOT = "F:/DOWN/sample-1000 -1"          # 原始4类文件夹
OUT_ROOT = "./dataset_split"        # 输出目录

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

IMG_EXTS = [".jpg", ".png", ".jpeg"]


# ==============================
# UTILS
# ==============================

def mkdir(p):
    os.makedirs(p, exist_ok=True)


def copy_pair(img_path, xml_path, img_dst, xml_dst):
    shutil.copy(img_path, img_dst)
    shutil.copy(xml_path, xml_dst)


# ==============================
# MAIN
# ==============================

def main():

    random.seed(42)

    splits = ["train", "val", "test"]

    for s in splits:
        mkdir(Path(OUT_ROOT) / s / "images")
        mkdir(Path(OUT_ROOT) / s / "annotations")

    # 遍历每个类别文件夹
    for cls in sorted(os.listdir(RAW_ROOT)):

        cls_dir = Path(RAW_ROOT) / cls
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

        n_train = int(total * TRAIN_RATIO)
        n_val = int(total * VAL_RATIO)

        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]

        print(f"  total={total}, train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}")

        # 复制函数
        def process(split_name, img_list):

            for img_path in img_list:

                xml_path = img_path.with_suffix(".xml")

                if not xml_path.exists():
                    print(f"⚠ Missing XML: {xml_path}")
                    continue

                dst_img = Path(OUT_ROOT) / split_name / "images" / img_path.name
                dst_xml = Path(OUT_ROOT) / split_name / "annotations" / xml_path.name

                copy_pair(img_path, xml_path, dst_img, dst_xml)

        process("train", train_imgs)
        process("val", val_imgs)
        process("test", test_imgs)

    print("\n✅ Stratified split finished!")


if __name__ == "__main__":
    main()
