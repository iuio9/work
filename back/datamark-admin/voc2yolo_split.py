"""
VOC XML → YOLO txt 转换工具

将 split.py 输出的 dataset_split 目录（包含图像和VOC格式XML标注）
转换为 YOLO 格式数据集，可直接被 YOLOv8/Faster R-CNN 训练脚本使用。

用法示例：
    # 使用自动类别映射（按类别名首次出现顺序）
    python voc2yolo_split.py --src_root ./dataset_split --dst_root ./dataset_yolo

    # 指定类别映射（JSON格式）
    python voc2yolo_split.py --src_root ./dataset_split --dst_root ./dataset_yolo \
        --class_map '{"cat":0,"dog":1}'

    # 从文件加载类别映射
    python voc2yolo_split.py --src_root ./dataset_split --dst_root ./dataset_yolo \
        --class_map_file ./classes.json

    # 同时生成 data.yaml（YOLOv8训练需要）
    python voc2yolo_split.py --src_root ./dataset_split --dst_root ./dataset_yolo \
        --write_yaml

输入目录结构：
    src_root/
      train/images/*.jpg
      train/annotations/*.xml
      val/images/*.jpg
      val/annotations/*.xml
      (test/...)

输出目录结构：
    dst_root/
      images/train/*.jpg
      images/val/*.jpg
      labels/train/*.txt
      labels/val/*.txt
      data.yaml  (如果 --write_yaml)
"""

import argparse
import json
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# =========================
# 默认配置
# =========================

DEFAULT_SRC_ROOT = "dataset_split"
DEFAULT_DST_ROOT = "dataset_yolo"

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


def discover_classes(src_root: Path) -> dict:
    """扫描所有XML自动构建类别映射"""
    class_names = set()
    for split in ("train", "val", "test"):
        ann_dir = src_root / split / "annotations"
        if not ann_dir.exists():
            continue
        for xml_path in ann_dir.glob("*.xml"):
            try:
                _, _, objs = parse_xml(xml_path)
                for name, *_ in objs:
                    class_names.add(name)
            except Exception as e:
                print(f"⚠ 解析XML失败 {xml_path}: {e}")

    sorted_names = sorted(class_names)
    return {name: idx for idx, name in enumerate(sorted_names)}


def process_split(src_root: Path, dst_root: Path, split: str, class_map: dict):
    img_src = src_root / split / "images"
    ann_src = src_root / split / "annotations"

    if not img_src.exists():
        print(f"⏭ 跳过不存在的 split: {split}")
        return 0

    img_dst = dst_root / "images" / split
    label_dst = dst_root / "labels" / split

    img_dst.mkdir(parents=True, exist_ok=True)
    label_dst.mkdir(parents=True, exist_ok=True)

    count = 0
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
            if name not in class_map:
                print(f"⚠ 未知类别 {name} in {xml_path}")
                continue

            cls_id = class_map[name]
            xc, yc, bw, bh = convert_box(w, h, xmin, xmax, ymin, ymax)
            yolo_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        # 拷贝图片 + 写入 label
        shutil.copy(img_path, img_dst / img_path.name)
        with open(label_dst / f"{img_path.stem}.txt", "w") as f:
            f.write("\n".join(yolo_lines))
        count += 1

    print(f"  ✓ {split}: {count} 个样本")
    return count


def parse_args():
    parser = argparse.ArgumentParser(description='VOC XML → YOLO txt 转换工具')
    parser.add_argument('--src_root', type=str, default=DEFAULT_SRC_ROOT,
                        help='源目录（split.py 输出）')
    parser.add_argument('--dst_root', type=str, default=DEFAULT_DST_ROOT,
                        help='目标YOLO数据集目录')
    parser.add_argument('--class_map', type=str, default=None,
                        help='类别映射JSON字符串，如 \'{"cat":0,"dog":1}\'')
    parser.add_argument('--class_map_file', type=str, default=None,
                        help='类别映射JSON文件路径')
    parser.add_argument('--write_yaml', action='store_true',
                        help='是否同时生成YOLOv8需要的 data.yaml')
    return parser.parse_args()


def main():
    args = parse_args()

    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)

    if not src_root.exists():
        raise FileNotFoundError(f"源目录不存在: {src_root}")

    # 加载/自动发现类别映射
    if args.class_map_file:
        with open(args.class_map_file, 'r', encoding='utf-8') as f:
            class_map = json.load(f)
    elif args.class_map:
        class_map = json.loads(args.class_map)
    else:
        print("🔍 未提供类别映射，自动扫描XML发现类别...")
        class_map = discover_classes(src_root)
        if not class_map:
            raise ValueError(f"未在 {src_root} 中发现任何类别")

    print(f"\n📋 类别映射 ({len(class_map)} 类):")
    for name, cid in sorted(class_map.items(), key=lambda x: x[1]):
        print(f"   {cid}: {name}")

    dst_root.mkdir(parents=True, exist_ok=True)

    total = 0
    for split in ("train", "val", "test"):
        print(f"\n📂 处理 {split} ...")
        total += process_split(src_root, dst_root, split, class_map)

    # 生成 data.yaml（YOLOv8训练需要）
    if args.write_yaml:
        try:
            import yaml
        except ImportError:
            print("⚠ PyYAML 未安装，跳过 data.yaml 生成: pip install pyyaml")
        else:
            # 按 id 排序得到 names 列表
            id_to_name = {cid: name for name, cid in class_map.items()}
            names = [id_to_name[i] for i in sorted(id_to_name.keys())]
            data_cfg = {
                'path': str(dst_root.resolve()),
                'train': 'images/train',
                'val': 'images/val',
                'test': 'images/test',
                'nc': len(names),
                'names': names,
            }
            yaml_path = dst_root / 'data.yaml'
            with open(yaml_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(data_cfg, f, default_flow_style=False, allow_unicode=True)
            print(f"\n📝 已生成 YOLOv8 配置: {yaml_path}")

    print(f"\n✅ 全部转换完成，共处理 {total} 个样本")
    print(f"   输出目录: {dst_root}")


if __name__ == "__main__":
    main()
