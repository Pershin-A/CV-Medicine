"""Train the complete original-image DXA localization pipeline.

The smoke mode runs one or two batches per task and checks the entire path;
its accuracy is intentionally not a model-quality estimate.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from dxa_project.geometry_ml.data import (DxaDataset, collate,
                                          load_augmented_records, load_records)
from dxa_project.geometry_ml.models import (
    Router, SpatialNet, artifact_detector, detector_images, spatial_loss,
)


TASKS = ("router", "spine", "hip", "artifact")


def split_records(records, fold: int):
    groups = np.array([r.study for r in records])
    if len(set(groups)) < 5:
        raise ValueError("At least five studies are required")
    splits = list(GroupKFold(n_splits=5).split(np.zeros(len(records)), groups=groups))
    train_ids, valid_ids = splits[fold]
    train = [records[i] for i in train_ids]
    valid = [records[i] for i in valid_ids]
    if {r.study for r in train} & {r.study for r in valid}:
        raise AssertionError("Study leakage between training and validation")
    return train, valid


def _subset(records, task: str, smoke: bool):
    if task in ("spine", "artifact"):
        records = [r for r in records if r.region == "SPINE"]
    elif task == "hip":
        records = [r for r in records if r.region.startswith("LEG_")]
    if not smoke:
        return records
    if task == "artifact":
        positives = []
        negatives = []
        for record in records:
            geometry = json.loads(record.geometry_path.read_text(encoding="utf-8"))
            (positives if geometry["spine"]["foreign_objects"] else negatives).append(record)
        return positives[:4] + negatives[:4]
    return records[:8]


def _model(task: str, pretrained: bool, architecture='light'):
    if task == "router":
        return Router(pretrained)
    if task in ("spine", "hip"):
        return SpatialNet("SPINE" if task == "spine" else "HIP", pretrained,
                          'resnet50' if architecture=='heavy' else 'resnet18')
    return artifact_detector(pretrained,heavy=architecture=='heavy')


def _loss(task, model, images, targets, device):
    images = images.to(device)
    if task == "router":
        truth = torch.stack([t["region"] for t in targets]).to(device)
        outputs = model(images)
        return nn.functional.cross_entropy(outputs, truth), outputs
    if task in ("spine", "hip"):
        outputs = model(images)
        return spatial_loss(outputs, targets, "SPINE" if task == "spine" else "HIP"), outputs
    boxes = [{"boxes": t["artifact_boxes"].to(device),
              "labels": torch.ones(len(t["artifact_boxes"]), dtype=torch.int64,
                                   device=device)} for t in targets]
    losses = model(detector_images(images), boxes)
    return sum(losses.values()), losses


def _preview(task, images, targets, output, path: Path):
    mean = torch.tensor((.485, .456, .406))[:, None, None]
    std = torch.tensor((.229, .224, .225))[:, None, None]
    image = ((images[0].cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    base = Image.fromarray(image)
    target = base.copy()
    predicted = base.copy()
    td, pd = ImageDraw.Draw(target, "RGBA"), ImageDraw.Draw(predicted, "RGBA")
    if task in ("spine", "hip"):
        key = "line" if task == "spine" else "trochanter"
        channel = 0 if task == "spine" else 3
        truth = targets[0][key][0].numpy() > .5
        probability = torch.sigmoid(output["spatial"][0, channel]).detach().cpu().numpy() > .5
        for draw, mask in ((td, truth), (pd, probability)):
            ys, xs = np.where(mask)
            for x, y in zip(xs[::2], ys[::2]):
                draw.point((int(x), int(y)), fill=(255, 55, 55, 210))
    elif task == "artifact":
        for x1, y1, x2, y2 in targets[0]["artifact_boxes"].tolist():
            td.rectangle((x1, y1, x2, y2), outline=(255, 55, 55, 255), width=2)
        for box, score in zip(output[0]["boxes"].detach().cpu(),
                              output[0]["scores"].detach().cpu()):
            if score >= .5:
                pd.rectangle(tuple(box.tolist()), outline=(255, 55, 55, 255), width=2)
    board = Image.new("RGB", (base.width * 3, base.height), "black")
    for i, part in enumerate((base, target, predicted)):
        board.paste(part, (i * base.width, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    board.save(path)


def _box_iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-9)


def _validation_stats(task, outputs, targets, size: int) -> dict:
    if task == "router":
        guesses = outputs.argmax(1).cpu()
        truth = torch.stack([t["region"] for t in targets])
        return {"correct": int((guesses == truth).sum()), "images": len(targets)}
    if task in ("spine", "hip"):
        probs = torch.sigmoid(outputs["spatial"]).cpu().numpy()
        key, channel = ("line", 0) if task == "spine" else ("trochanter", 3)
        stats = {"intersection": 0, "predicted_pixels": 0, "target_pixels": 0,
                 "point_error_sum_px": 0.0,"point_error_sum_mm":0., "visible_points": 0}
        point_key = "crest" if task == "spine" else "hip_points"
        present_key = "crest_present" if task == "spine" else "hip_present"
        first_channel = 1 if task == "spine" else 0
        for n, target in enumerate(targets):
            pred = probs[n, channel] >= .5
            truth = target[key][0].numpy() >= .5
            stats["intersection"] += int((pred & truth).sum())
            stats["predicted_pixels"] += int(pred.sum())
            stats["target_pixels"] += int(truth.sum())
            for k, visible in enumerate(target[present_key]):
                if visible < .5:
                    continue
                py, px = np.unravel_index(probs[n, first_channel + k].argmax(),
                                          (size, size))
                ty, tx = np.unravel_index(target[point_key][k].numpy().argmax(),
                                          (size, size))
                stats["point_error_sum_px"] += float(np.hypot(px - tx, py - ty))
                row_mm,col_mm=target['spacing_mm']
                stats['point_error_sum_mm']+=float(np.hypot((px-tx)/target['scale']*col_mm,(py-ty)/target['scale']*row_mm))
                stats["visible_points"] += 1
        if task == "hip":
            stats.update({"roi_iou_sum": 0.0, "roi_count": 0})
            for n, target in enumerate(targets):
                if not target["roi_present"].item():
                    continue
                y1, y2, x2 = outputs["roi"][n].detach().cpu().numpy().tolist()
                gt_y1, gt_y2, gt_x2 = target["roi"].tolist()
                x1 = target["pad_left"] / size
                stats["roi_iou_sum"] += _box_iou(
                    (x1, min(y1, y2), x2, max(y1, y2)),
                    (x1, gt_y1, gt_x2, gt_y2))
                stats["roi_count"] += 1
        return stats
    stats = {"matched_boxes": 0, "ground_truth_boxes": 0,
             "predicted_boxes": 0, "images": len(targets)}
    for prediction, target in zip(outputs, targets):
        boxes = prediction["boxes"].detach().cpu().numpy()
        scores = prediction["scores"].detach().cpu().numpy()
        chosen = boxes[scores >= .5]
        truth = target["artifact_boxes"].numpy()
        stats["ground_truth_boxes"] += len(truth)
        stats["predicted_boxes"] += len(chosen)
        available = set(range(len(chosen)))
        for box in truth:
            if available:
                best = max(available, key=lambda k: _box_iou(box, chosen[k]))
                if _box_iou(box, chosen[best]) >= .5:
                    stats["matched_boxes"] += 1
                    available.remove(best)
    return stats


def _aggregate_stats(rows: list[dict], task: str) -> dict:
    totals = {key: sum(row[key] for row in rows) for key in rows[0]}
    if task == "router":
        return {"accuracy": totals["correct"] / totals["images"], **totals}
    if task in ("spine", "hip"):
        totals["pixel_dice"] = (2 * totals["intersection"] /
                                max(totals["predicted_pixels"] + totals["target_pixels"], 1))
        totals["mean_point_error_model_px"] = (totals["point_error_sum_px"] /
                                                max(totals["visible_points"], 1))
        totals['mean_point_error_nominal_mm']=totals['point_error_sum_mm']/max(totals['visible_points'],1)
        totals['pixel_iou']=totals['intersection']/max(totals['predicted_pixels']+totals['target_pixels']-totals['intersection'],1)
        if task == "hip":
            totals["mean_roi_iou"] = totals["roi_iou_sum"] / max(totals["roi_count"], 1)
        return totals
    totals["box_recall_iou50"] = totals["matched_boxes"] / max(totals["ground_truth_boxes"], 1)
    totals["box_precision_iou50"] = totals["matched_boxes"] / max(totals["predicted_boxes"], 1)
    return totals


def train_task(task: str, train_records, valid_records, output: Path, device,
               epochs: int, size: int, batch_size: int, max_batches: int | None,
               pretrained: bool, architecture='light', loader_workers=0) -> dict:
    train_records = _subset(train_records, task, max_batches is not None)
    valid_records = _subset(valid_records, task, max_batches is not None)
    if not train_records or not valid_records:
        raise ValueError(f"No training or validation records for {task}")
    sampler=None
    if task=='artifact':
        labels=[int(bool(json.loads(r.geometry_path.read_text(encoding='utf-8'))['spine']['foreign_objects'])) for r in train_records]
        counts=np.bincount(labels,minlength=2)
        if np.all(counts>0):
            sampler=WeightedRandomSampler([1/float(counts[label]) for label in labels],len(labels),replacement=True)
    train_loader = DataLoader(DxaDataset(train_records, size, router_mode=task == "router"), batch_size=batch_size,
                              shuffle=sampler is None,sampler=sampler, num_workers=loader_workers, collate_fn=collate,
                              persistent_workers=loader_workers>0,pin_memory=device.type=='cuda')
    valid_loader = DataLoader(DxaDataset(valid_records, size, router_mode=task == "router"), batch_size=batch_size,
                              shuffle=False, num_workers=loader_workers, collate_fn=collate,
                              persistent_workers=loader_workers>0,pin_memory=device.type=='cuda')
    model = _model(task, pretrained,architecture).to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    learning_rate = (1e-3 if task == "router" else
                     3e-5 if task == "artifact" else 2e-4)
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate,
                                  weight_decay=1e-4)
    history = []
    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for step, (images, targets) in enumerate(train_loader):
            if max_batches is not None and step >= max_batches:
                break
            optimizer.zero_grad(set_to_none=True)
            loss, details = _loss(task, model, images, targets, device)
            if not torch.isfinite(loss):
                detail_values = ({key: float(value.detach().cpu()) for key, value in details.items()}
                                 if task == "artifact" else {})
                raise FloatingPointError(
                    f"Non-finite {task} loss, epoch={epoch + 1}, step={step}, "
                    f"images={[t['relative_path'] for t in targets]}, "
                    f"parts={detail_values}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=5.0,
                                           error_if_nonfinite=True)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        valid_losses = []
        validation_stats = []
        with torch.no_grad():
            for step, (images, targets) in enumerate(valid_loader):
                if max_batches is not None and step >= max_batches:
                    break
                if task == "artifact":
                    predictions = model(detector_images(images.to(device)))
                    valid_losses.append(float(len(predictions[0]["boxes"])))
                    validation_stats.append(_validation_stats(task, predictions, targets, size))
                    if epoch == epochs - 1:
                        _preview(task, images, targets, predictions, output / f"{task}_preview.png")
                else:
                    loss, outputs = _loss(task, model, images, targets, device)
                    valid_losses.append(float(loss.detach().cpu()))
                    validation_stats.append(_validation_stats(task, outputs, targets, size))
                    if epoch == epochs - 1 and task in ("spine", "hip"):
                        _preview(task, images, targets, outputs, output / f"{task}_preview.png")
        if device.type == "cuda":
            torch.cuda.synchronize()
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(train_losses)),
                        "validation_loss_or_detection_count": float(np.mean(valid_losses)),
                        "validation_metrics": _aggregate_stats(validation_stats, task)})
        print(json.dumps({'task':task,'epoch':epoch+1,'elapsed_seconds':time.perf_counter()-start,
                          **history[-1]},ensure_ascii=False),flush=True)
        (output/f'{task}_history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        torch.save({'task':task,'size':size,'state_dict':model.state_dict(),
                    'pretrained':pretrained,'architecture':architecture,'epoch':epoch+1},output/f'{task}.pt')
    elapsed = time.perf_counter() - start
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"task": task, "size": size, "state_dict": model.cpu().state_dict(),
                "pretrained": pretrained,'architecture':architecture}, output / f"{task}.pt")
    return {"task": task, "train_images": len(train_records),
            "validation_images": len(valid_records), "epochs": epochs,
            "train_batches_per_epoch": len(train_losses), "seconds": elapsed,
            "history": history,
            'artifact_class_balancing':sampler is not None,
            'empty_negative_roi_batches':getattr(model.roi_heads, 'empty_negative_batches', 0) if task == 'artifact' else None,
            "preview": str(output / f"{task}_preview.png") if task != "router" else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--task", choices=(*TASKS, "all"), default="all")
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument('--architecture',choices=('light','heavy'),default='light')
    parser.add_argument('--loader-workers',type=int,default=0)
    parser.add_argument("--augmented-root", type=Path,
                        help="Optional generator output; only train-side sources are added")
    parser.add_argument('--augmented-annotations-root',type=Path,help='Optional manual annotation overlay with labels.csv; keeps base generated JSON unchanged')
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output or root / "dxa_project" / "outputs" /
              ("geometry_ml_smoke" if args.smoke else "geometry_ml_train")).resolve()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.hub.set_dir(str(root / "dxa_project" / "outputs" / "torch_hub"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision('high')
    if device.type=='cuda':
        torch.backends.cudnn.benchmark=True
    records = load_records(root)
    train_records, valid_records = split_records(records, args.fold)
    augmented_count = 0
    if args.augmented_root is not None:
        augmented = load_augmented_records(root, args.augmented_root.resolve(), records,args.augmented_annotations_root)
        train_studies = {record.study for record in train_records}
        train_records.extend(record for record in augmented if record.study in train_studies)
        augmented_count = len(train_records) - (len(records) - len(valid_records))
        if {record.study for record in train_records} & {record.study for record in valid_records}:
            raise AssertionError("Augmented study leaked into validation")
    output.mkdir(parents=True, exist_ok=True)
    report = {"device": str(device), "torch": torch.__version__, "fold": args.fold,
              "train_studies": len({r.study for r in train_records}),
              "validation_studies": len({r.study for r in valid_records}),
              "smoke": args.smoke, "augmentation": bool(args.augmented_root),
              "augmented_train_images": augmented_count, "tasks": []}
    report['manual_augmented_annotation_overlay']=str(args.augmented_annotations_root) if args.augmented_annotations_root else None
    for task in TASKS if args.task == "all" else (args.task,):
        result = train_task(task, train_records, valid_records, output, device,
                            1 if args.smoke else args.epochs,
                            args.size, args.batch_size,
                            2 if args.smoke else None,
                            not args.random_init,args.architecture,args.loader_workers)
        report["tasks"].append(result)
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False), flush=True)
    print(json.dumps({"report": str(output / "report.json"),
                      "device": str(device)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
