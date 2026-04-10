import os
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil

# =========================
# 配置
# =========================

SRC_ROOT = "dataset_split"   # 你现在的数据
DST_ROOT = "dataset_yolo"    # 转成 YOLO 后

CLASS_MAP = {
    "021_tdhj_xxsgwpyh_dz": 0,
    "021_tdhj_xxsgwpyh_wjj": 1,
    "021_tdhj_xxshywyh_sh-yw_sh": 2,
    "021_tdhj_xxshywyh_sh-yw_yw": 3
}

IMG_SUFFIX = [".jpg", ".png", ".jpeg"]

# =========================


def convert_box(img_w, img_h, xmin, xmax, ymin, ymax):
    xc = (xmin + xmax) / 2.0 / img_w
    yc = (ymin + ymax) / 2.0 / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    return xc, yc, w, h


def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()

        box = obj.find("bndbox")
        xmin = int(box.find("xmin").text)
        xmax = int(box.find("xmax").text)
        ymin = int(box.find("ymin").text)
        ymax = int(box.find("ymax").text)

        objects.append((name, xmin, xmax, ymin, ymax))

    return w, h, objects


# =========================

def process_split(split):

    img_src = Path(SRC_ROOT, split, "images")
    ann_src = Path(SRC_ROOT, split, "annotations")

    img_dst = Path(DST_ROOT, "images", split)
    label_dst = Path(DST_ROOT, "labels", split)

    img_dst.mkdir(parents=True, exist_ok=True)
    label_dst.mkdir(parents=True, exist_ok=True)

    for img_path in img_src.iterdir():

        if img_path.suffix.lower() not in IMG_SUFFIX:
            continue

        xml_path = ann_src / f"{img_path.stem}.xml"

        if not xml_path.exists():
            print(f"⚠ 缺少标注: {img_path.name}")
            continue

        w, h, objects = parse_xml(xml_path)

        yolo_lines = []

        for name, xmin, xmax, ymin, ymax in objects:

            if name not in CLASS_MAP:
                print(f"⚠ 未知类别 {name} in {xml_path}")
                continue

            cls_id = CLASS_MAP[name]

            xc, yc, bw, bh = convert_box(
                w, h,
                xmin, xmax,
                ymin, ymax
            )

            yolo_lines.append(
                f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
            )

        # 拷贝图片
        shutil.copy(img_path, img_dst / img_path.name)

        # 写入 label
        with open(label_dst / f"{img_path.stem}.txt", "w") as f:
            f.write("\n".join(yolo_lines))


# =========================

def main():

    for split in ["train", "val", "test"]:
        print(f"📂 处理 {split} ...")
        process_split(split)

    print("\n✅ 全部转换完成，YOLOv8 可直接使用！")


if __name__ == "__main__":
    main()
