"""One real RF-DETR fine-tuning job, run as a subprocess by services/trainer_rfdetr.py on the GPU pod.

  python train_job.py <job dir>        (the job's settings are in <job dir>/job.json)

Stages, written to <job dir>/status.json as they happen (the service reads them):
  downloading dataset → training → evaluating → done   (or failed, with the error)
<job dir>/log.txt gets one line per step and per epoch ("epoch 3/10  loss 4.12  val mAP50 0.31", the format the web
app's chart reads); RF-DETR's own output goes to <job dir>/rfdetr.log. The result (weights, args, per-class scores)
goes to <job dir>/result.json.

The dataset is the gateway's COCO dataset (Roboflow layout: <rid>/train, <rid>/valid) in a private Hugging Face
dataset repo, read with HF_TOKEN (read-only is enough).
"""

import csv
import hashlib
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path

SIZES = ("nano", "small", "medium", "large")
VALID_SHARE = 5  # with no valid split, every 5th training image (20 %) is held back for validation
IOU = 0.5  # a prediction matches a box at IoU >= 0.5
CONF = 0.25  # scored at the gateway's pre-label threshold: what pre-labelling with these weights would draw


class Job:
    def __init__(self, root):
        self.root = Path(root)
        self.cfg = json.loads((self.root / "job.json").read_text())
        self.state = {"status": "running", "stage": "downloading dataset", "progress": 0.0, "error": None}

    def set(self, **kw):
        self.state.update(kw)
        tmp = self.root / "status.json.tmp"
        tmp.write_text(json.dumps(self.state))
        tmp.replace(self.root / "status.json")

    def log(self, line):
        with open(self.root / "log.txt", "a") as f:
            f.write(line.rstrip() + "\n")


# ---- dataset ------------------------------------------------------------------------------------------------------
def parse_location(location):
    """hf://<org>/<repo>/<rid> → (repo id, rid). Anything else is refused: the pod only reads the Hub."""
    if not location.startswith("hf://"):
        raise ValueError(f"the training pod reads datasets from Hugging Face (hf://org/repo/rid), not {location!r}")
    parts = location.removeprefix("hf://").strip("/").split("/")
    if len(parts) != 3 or not all(parts):
        raise ValueError(f"expected hf://<org>/<repo>/<dataset id>, got {location!r}")
    return f"{parts[0]}/{parts[1]}", parts[2]


def download(location, dest):
    from huggingface_hub import snapshot_download
    repo, rid = parse_location(location)
    snapshot_download(repo_id=repo, repo_type="dataset", allow_patterns=[f"{rid}/*", f"{rid}/**"], local_dir=str(dest),
                      token=os.environ.get("HF_TOKEN") or None)
    root = Path(dest) / rid
    if not (root / "train" / "_annotations.coco.json").exists():
        raise ValueError(f"{location} has no train/_annotations.coco.json")
    return root


def load_coco(split_dir):
    p = Path(split_dir) / "_annotations.coco.json"
    return json.loads(p.read_text()) if p.exists() else None


def classes_of(coco):
    return [c["name"] for c in sorted(coco["categories"], key=lambda c: c["id"])]


def ensure_valid(root, log):
    """RF-DETR needs a valid split. A dataset from a short video has none: hold back every 5th training image (the
    scores are then optimistic, the frames being near the training ones). Returns True when it did."""
    root = Path(root)
    valid = load_coco(root / "valid")
    if valid and valid["images"]:
        return False
    train = load_coco(root / "train")
    images = sorted(train["images"], key=lambda i: i["file_name"])
    if len(images) < 2:
        raise ValueError("the dataset needs at least 2 images")
    held = {i["id"] for k, i in enumerate(images) if k % VALID_SHARE == VALID_SHARE - 1} or {images[-1]["id"]}
    (root / "valid").mkdir(exist_ok=True)
    split = {"train": {"images": [], "annotations": []}, "valid": {"images": [], "annotations": []}}
    for img in train["images"]:
        side = "valid" if img["id"] in held else "train"
        split[side]["images"].append(img)
        if side == "valid":
            shutil.move(str(root / "train" / img["file_name"]), str(root / "valid" / img["file_name"]))
    for a in train["annotations"]:
        split["valid" if a["image_id"] in held else "train"]["annotations"].append(a)
    for side, d in split.items():
        (root / side / "_annotations.coco.json").write_text(json.dumps({**train, **d}))
    log(f"no valid split in the dataset: {len(held)} of {len(images)} training images held back for validation "
        f"(scores are optimistic: these frames are close to the training ones)")
    return True


def cover_valid(root, log):
    """A class with training boxes but none in valid is never scored, nor counted when the best checkpoint is picked:
    move every 5th training image with it (at least one) to valid. Returns the classes it did that for."""
    root = Path(root)
    train, valid = load_coco(root / "train"), load_coco(root / "valid")
    names = {c["id"]: c["name"] for c in train["categories"]}
    in_valid = {a["category_id"] for a in valid["annotations"]}
    held, covered = set(), []
    for cid in sorted({a["category_id"] for a in train["annotations"]} - in_valid):
        with_it = sorted({a["image_id"] for a in train["annotations"] if a["category_id"] == cid} - held)
        if len(with_it) < 2:
            continue  # its only image stays for training
        held |= {i for k, i in enumerate(with_it) if k % VALID_SHARE == VALID_SHARE - 1} or {with_it[-1]}
        covered.append(names.get(cid, str(cid)))
    if not held:
        return []
    for img in train["images"]:
        if img["id"] in held:
            shutil.move(str(root / "train" / img["file_name"]), str(root / "valid" / img["file_name"]))
    # image and annotation ids are per file: renumber the moved ones past valid's own
    img_base = max((i["id"] for i in valid["images"]), default=-1) + 1
    ann_base = max((a["id"] for a in valid["annotations"]), default=-1) + 1
    new_id = {i: img_base + k for k, i in enumerate(sorted(held))}
    moved = [a for a in train["annotations"] if a["image_id"] in held]
    valid["images"] += [{**i, "id": new_id[i["id"]]} for i in train["images"] if i["id"] in held]
    valid["annotations"] += [{**a, "id": ann_base + k, "image_id": new_id[a["image_id"]]} for k, a in enumerate(moved)]
    train["images"] = [i for i in train["images"] if i["id"] not in held]
    train["annotations"] = [a for a in train["annotations"] if a["image_id"] not in held]
    for side, d in (("train", train), ("valid", valid)):
        (root / side / "_annotations.coco.json").write_text(json.dumps(d))
    log(f"no {', '.join(covered)} in the valid split: {len(held)} training images with it held back for validation "
        f"(its scores are optimistic: these frames are close to the training ones)")
    return covered


# ---- metrics ------------------------------------------------------------------------------------------------------
def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def per_class_scores(classes, frames):
    """frames: [(truth [(class, xyxy)], predicted [(class, conf, xyxy)])] → {class: {recall, precision}} in %, greedy
    matching per class at IoU >= 0.5 (predictions below CONF are ignored). A class with no boxes at all scores 0."""
    tp = dict.fromkeys(classes, 0)
    n_true, n_pred = dict.fromkeys(classes, 0), dict.fromkeys(classes, 0)
    for truth, pred in frames:
        for c in classes:
            t = [b for k, b in truth if k == c]
            p = sorted(((s, b) for k, s, b in pred if k == c and s >= CONF), key=lambda x: -x[0])
            n_true[c] += len(t)
            n_pred[c] += len(p)
            used = set()
            for _, b in p:
                best = max(((iou(b, tb), j) for j, tb in enumerate(t) if j not in used), default=(0.0, None))
                if best[0] >= IOU:
                    used.add(best[1])
                    tp[c] += 1
    return {c: {"recall": round(100 * tp[c] / n_true[c]) if n_true[c] else 0,
                "precision": round(100 * tp[c] / n_pred[c]) if n_pred[c] else 0} for c in classes}


def epoch_lines(metrics_csv, epochs):
    """RF-DETR's metrics.csv (one row per logged step / epoch) → [(epoch, loss, val mAP50)] for finished epochs."""
    p = Path(metrics_csv)
    if not p.exists():
        return []
    rows = {}
    with open(p, newline="") as f:
        for r in csv.DictReader(f):
            if not r.get("epoch"):
                continue
            e = int(float(r["epoch"])) + 1
            d = rows.setdefault(e, {})
            for k in ("train/loss_epoch", "train/loss"):
                if r.get(k):
                    d.setdefault("loss", float(r[k]))
            if r.get("val/mAP_50"):
                d["map"] = float(r["val/mAP_50"])
    # an epoch counts once both its train loss and its val mAP50 are written (they come in separate rows)
    return [(e, d["loss"], d["map"]) for e, d in sorted(rows.items()) if "map" in d and "loss" in d and e <= epochs]


# ---- the job ------------------------------------------------------------------------------------------------------
def sha256_of(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def model_class(size):
    import rfdetr
    return {"nano": rfdetr.RFDETRNano, "small": rfdetr.RFDETRSmall, "medium": rfdetr.RFDETRMedium,
            "large": rfdetr.RFDETRLarge}[size]


def read_lines(job, prefix):
    p = job.root / "log.txt"
    return [line for line in p.read_text().splitlines() if line.startswith(prefix)] if p.exists() else []


def watch_epochs(job, out, epochs, stop):
    """Turn RF-DETR's metrics.csv into log lines and progress while it trains."""
    seen = 0
    while not stop.is_set():
        lines = epoch_lines(out / "metrics.csv", epochs)
        for e, loss, m in lines[seen:]:
            job.log(f"epoch {e}/{epochs}  loss {loss:.2f}  val mAP50 {m:.2f}")
            job.set(progress=round(0.1 + 0.8 * e / epochs, 3), epoch=e)
        seen = len(lines)
        stop.wait(5)


def evaluate(job, weights, root, classes):
    import numpy as np
    from PIL import Image
    cfg = job.cfg
    model = model_class(cfg["model"])(pretrain_weights=str(weights))
    coco = load_coco(root / "valid")
    by_image = {}
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        by_image.setdefault(a["image_id"], []).append((classes[a["category_id"]], [x, y, x + w, y + h]))
    frames = []
    for img in coco["images"]:
        rgb = np.array(Image.open(root / "valid" / img["file_name"]).convert("RGB"))
        d = model.predict(rgb, threshold=CONF)
        pred = [(classes[int(k)], float(s), [float(v) for v in b])
                for b, s, k in zip(d.xyxy, d.confidence, d.class_id) if int(k) < len(classes)]
        frames.append((by_image.get(img["id"], []), pred))
    return per_class_scores(classes, frames), len(frames)


def run(job):
    cfg = job.cfg
    if cfg["model"] not in SIZES:
        raise ValueError(f"model size must be one of {', '.join(SIZES)}")
    job.log(f"{cfg['id']} on pod {cfg.get('pod_id') or '-'} · RFDETR{cfg['model'].title()} · {cfg['epochs']} epochs")
    job.set(stage="downloading dataset", progress=0.02)
    job.log(f"downloading {cfg['dataset']} with the pod's HF token")
    root = download(cfg["dataset"], job.root / "data")
    classes = classes_of(load_coco(root / "train"))
    if cfg.get("classes") and list(cfg["classes"]) != classes:
        job.log(f"note: the dataset's classes {classes} differ from the job's {cfg['classes']}: using the dataset's")
    held_back = ensure_valid(root, job.log)
    covered = [] if held_back else cover_valid(root, job.log)
    n ={s: len(load_coco(root / s)["images"]) for s in ("train", "valid")}
    job.log(f"dataset: {n['train']} train · {n['valid']} valid images · {len(classes)} classes: {', '.join(classes)}")

    job.set(stage="training", progress=0.1, epoch=0)
    out = job.root / "out"
    stop = threading.Event()
    watcher = threading.Thread(target=watch_epochs, args=(job, out, cfg["epochs"], stop), daemon=True)
    watcher.start()
    model = model_class(cfg["model"])()
    try:
        model.train(dataset_dir=str(root), epochs=cfg["epochs"], batch_size=cfg.get("batch_size", 4),
                    grad_accum_steps=cfg.get("grad_accum_steps", 4), output_dir=str(out))
    finally:
        stop.set()
        watcher.join(10)
    done_epochs = epoch_lines(out / "metrics.csv", cfg["epochs"])
    for e, loss, m in done_epochs[len(read_lines(job, "epoch ")):]:  # the last epochs the watcher didn't see yet
        job.log(f"epoch {e}/{cfg['epochs']}  loss {loss:.2f}  val mAP50 {m:.2f}")
    if done_epochs:
        job.log(f"best checkpoint saved (best val mAP50 {max(m for _, _, m in done_epochs):.2f})")
    weights = out / "checkpoint_best_total.pth"
    if not weights.exists():
        raise RuntimeError("training ended without checkpoint_best_total.pth")

    job.set(stage="evaluating", progress=0.92)
    scores, frames = evaluate(job, weights, root, classes)
    job.log(f"evaluated on {frames} valid images: " + " · ".join(
        f"{c} R{s['recall']} P{s['precision']}" for c, s in scores.items()))
    result = {"weights": str(weights), "sha256": sha256_of(weights), "bytes": weights.stat().st_size,
              "args": {"model": cfg["model"], "epochs": cfg["epochs"], "dataset": cfg["dataset"], "classes": classes,
                       "resolution": None, "job_id": cfg["id"]},
              "metrics": {"split": "valid (held back from train)" if held_back else
                          f"valid (+ held back from train for {', '.join(covered)})" if covered else "valid",
                          "classes": scores}}
    (job.root / "result.json").write_text(json.dumps(result))
    job.log(f"{cfg['id']} done → {cfg['result_model']}")
    job.set(status="done", stage="done", progress=1.0)


def main():
    job = Job(sys.argv[1])
    t0 = time.time()
    try:
        run(job)
    except Exception as e:  # noqa: BLE001 - every failure is reported to the gateway through status.json
        job.log(f"failed: {type(e).__name__}: {e}")
        job.set(status="failed", error=f"{type(e).__name__}: {e}")
        raise
    finally:
        job.log(f"took {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
