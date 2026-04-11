import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader

from pathlib import Path
import cv2
from tqdm import tqdm

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# =====================
# Config
# =====================

DATA_ROOT = Path("autodl-tmp/dataset_yolo")

NUM_CLASSES = 4 + 1  # + background

BATCH = 4
EPOCHS = 40
LR = 1e-4
DEVICE = "cuda"

SAVE_DIR = Path("./resnet_detector1")
SAVE_DIR.mkdir(exist_ok=True)


# =====================
# Dataset
# =====================

class YoloDataset(Dataset):

    def __init__(self, split="train"):

        self.img_dir = DATA_ROOT / "images" / split
        self.label_dir = DATA_ROOT / "labels" / split

        self.images = sorted(list(self.img_dir.glob("*.jpg")))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_path = self.images[idx]
        label_path = self.label_dir / f"{img_path.stem}.txt"

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        boxes = []
        labels = []

        if label_path.exists():

            with open(label_path) as f:

                for line in f:
                    cid, xc, yc, bw, bh = map(float, line.split())

                    x1 = (xc - bw / 2) * w
                    y1 = (yc - bh / 2) * h
                    x2 = (xc + bw / 2) * w
                    y2 = (yc + bh / 2) * h

                    boxes.append([x1, y1, x2, y2])
                    labels.append(int(cid) + 1)

        # ⚠️ FasterRCNN 不允许 empty target
        if len(boxes) == 0:

            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        else:

            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
        }

        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.

        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))


# =====================
# COCO Evaluator
# =====================

class CocoEvaluator:

    def __init__(self, dataset):

        self.dataset = dataset

        self.img_ids = []
        self.anns = []
        self.cats = []

        ann_id = 1

        for i in range(len(dataset)):

            _, target = dataset[i]

            self.img_ids.append({"id": i})

            for box, label in zip(target["boxes"], target["labels"]):

                x1, y1, x2, y2 = box.tolist()

                self.anns.append({
                    "id": ann_id,
                    "image_id": i,
                    "category_id": int(label),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1),
                    "iscrowd": 0,
                })

                ann_id += 1

        self.cats = [{"id": i} for i in range(1, NUM_CLASSES)]

        coco_dict = {
            "images": self.img_ids,
            "annotations": self.anns,
            "categories": self.cats,
        }

        self.coco_gt = COCO()
        self.coco_gt.dataset = coco_dict
        self.coco_gt.createIndex()

    def evaluate(self, model, dataloader):

        model.eval()

        coco_results = []

        with torch.no_grad():

            for imgs, targets in tqdm(dataloader, desc="Validating"):

                imgs = [i.to(DEVICE) for i in imgs]
                outputs = model(imgs)

                for out, tgt in zip(outputs, targets):

                    img_id = int(tgt["image_id"])

                    for box, score, label in zip(
                        out["boxes"],
                        out["scores"],
                        out["labels"]
                    ):

                        coco_results.append({
                            "image_id": img_id,
                            "category_id": int(label),
                            "bbox": [
                                box[0].item(),
                                box[1].item(),
                                box[2].item() - box[0].item(),
                                box[3].item() - box[1].item(),
                            ],
                            "score": score.item(),
                        })

        coco_dt = self.coco_gt.loadRes(coco_results)

        coco_eval = COCOeval(self.coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        return coco_eval.stats


# =====================
# Train + Val
# =====================

def train():

    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        NUM_CLASSES
    )

    model.to(DEVICE)

    train_set = YoloDataset("train")
    val_set = YoloDataset("val")

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )

    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )

    evaluator = CocoEvaluator(val_set)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_map = 0

    for epoch in range(EPOCHS):

        model.train()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for imgs, targets in pbar:

            # ===== skip empty targets =====
            valid_imgs = []
            valid_targets = []

            for img, tgt in zip(imgs, targets):

                if tgt["boxes"].shape[0] > 0:
                    valid_imgs.append(img)
                    valid_targets.append(tgt)

            if len(valid_imgs) == 0:
                continue

            imgs = [i.to(DEVICE) for i in valid_imgs]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in valid_targets]

            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=float(loss))

        # ---------- validation ----------
        stats = evaluator.evaluate(model, val_loader)

        map5095 = stats[0]
        map50 = stats[1]

        print(f"\n📊 Epoch {epoch}: mAP50={map50:.4f}, mAP50-95={map5095:.4f}")

        if map5095 > best_map:

            best_map = map5095
            torch.save(model.state_dict(), SAVE_DIR / "best.pth")
            print("✅ Best model saved")

        torch.save(model.state_dict(), SAVE_DIR / "last.pth")

    torch.save(model.state_dict(), SAVE_DIR / "final.pth")


# =====================
# Main
# =====================

if __name__ == "__main__":
    train()