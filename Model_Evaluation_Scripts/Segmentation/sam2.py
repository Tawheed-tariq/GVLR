import argparse
import json
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor
from scipy.ndimage import zoom
import ijson


# -----------------------------
# STREAM DATASET JSON (large)
# -----------------------------
def stream_json(path):
    with open(path, "r") as f:
        for obj in ijson.items(f, "item"):
            yield obj


# -----------------------------
# STREAM PREDICTIONS (JSONL or ARRAY)
# -----------------------------
def stream_predictions(path):
    with open(path, "r") as f:
        first_char = f.read(1)
        f.seek(0)

        # JSON array
        if first_char == "[":
            for obj in ijson.items(f, "item"):
                if isinstance(obj, dict):
                    yield obj
            return

        # JSONL
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                return
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj
            except json.JSONDecodeError as e:
                print(f"Skipping malformed line {i}: {e}")


# -----------------------------
def mask_iou(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return inter, union, inter / union if union > 0 else 0.0


# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--qwen_results", required=True)
    p.add_argument("--dataset_json", required=True)
    p.add_argument("--image_dir", required=True)
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


# -----------------------------
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("\nLoading SAM2 model...")
    predictor = SAM2ImagePredictor.from_pretrained(
        "facebook/sam2-hiera-large"
    )
    print("SAM2 loaded ✓")

    # ---------------------------------------------------
    # INDEX DATASET (single pass)
    # ---------------------------------------------------
    print("\nIndexing dataset...")
    dataset_map = {}

    for item in stream_json(args.dataset_json):
        dataset_map[item["image_id"]] = item

    print(f"Indexed {len(dataset_map)} items ✓")

    output_file = os.path.join(args.output_dir, "sam2_results.jsonl")

    total = 0
    success = 0
    iou_sum = 0.0

    with open(output_file, "w") as out_f:

        for pred in tqdm(stream_predictions(args.qwen_results),
                         desc="SAM2 processing"):

            total += 1
            image_id = pred.get("image_id")

            # if not pred.get("success", False):
            #     out_f.write(json.dumps({
            #         "image_id": image_id,
            #         "success": False,
            #         "error": "Qwen failed"
            #     }) + "\n")
                # return

            gt_item = dataset_map.get(image_id)

            if gt_item is None:
                out_f.write(json.dumps({
                    "image_id": image_id,
                    "success": False,
                    "error": "missing dataset entry"
                }) + "\n")
                return

            img_path = os.path.join(
                args.image_dir,
                gt_item["image"].lstrip("/")
            )

            if not os.path.exists(img_path):
                out_f.write(json.dumps({
                    "image_id": image_id,
                    "success": True,
                    "mask_score": 0, 
                    "mask_iou": 0,
                    "mask_inter": 0, 
                    "mask_union": 0
                }) + "\n")
                return

            try:
                img = Image.open(img_path).convert("RGB")
                img_np = np.array(img)
                h, w = img_np.shape[:2]

                predictor.set_image(img_np)

                # ----------------------------------------
                # PROMPT SELECTION
                # ----------------------------------------
                box = None
                point = None

                if "bbox" in pred:
                    box = pred["bbox"]
                elif "bbox_2d" in pred:
                    box = pred["bbox_2d"]

                if "point" in pred:
                    point = pred["point"]
                elif "point_2d" in pred:
                    point = pred["point_2d"]

                if box is None or point is None:
                    out_f.write(json.dumps({
                        "image_id": image_id,
                        "success": True,
                        "mask_score": 0, 
                        "mask_iou": 0,
                        "mask_inter": 0, 
                        "mask_union": 0
                    }) + "\n")
                    return

                if box == [0,0,0,0] or point == [0,0]:
                    out_f.write(json.dumps({
                        "image_id": image_id,
                        "success": True,
                        "mask_score": 0, 
                        "mask_iou": 0,
                        "mask_inter": 0, 
                        "mask_union": 0
                    }) + "\n")
                    continue

                masks, scores, _ = predictor.predict(
                    box=np.array(box, dtype=np.float32),
                    point_coords=np.array([point], dtype=np.float32),
                    point_labels=np.array([1], dtype=np.int32),
                )

                if masks is None or len(masks) == 0:
                    raise RuntimeError("no mask generated")

                best_idx = int(np.argmax(scores))
                pred_mask = masks[best_idx].astype(bool)

                # resize if needed
                if pred_mask.shape[:2] != (h, w):
                    scale_y = h / pred_mask.shape[0]
                    scale_x = w / pred_mask.shape[1]
                    pred_mask = zoom(
                        pred_mask,
                        (scale_y, scale_x),
                        order=0
                    ).astype(bool)

                gt_mask = np.array(gt_item["mask"], dtype=bool)

                inter, union, iou = mask_iou(pred_mask, gt_mask)

                success += 1
                iou_sum += iou

                result = {
                    "image_id": image_id,
                    "success": True,
                    "mask_score": float(scores[best_idx]),
                    "mask_iou": float(iou),
                    "mask_inter": int(inter),
                    "mask_union": int(union)
                }

                out_f.write(json.dumps(result) + "\n")
                out_f.flush()

                torch.cuda.empty_cache()

            except Exception as e:
                print(str(e))
                out_f.write(json.dumps({
                    "image_id": image_id,
                    "success": False,
                    "error": str(e)
                }) + "\n")
                return

    print("\nFinished ✓")
    print("Processed:", total)
    print("Successful:", success)
    if success:
        print("Average IoU:", iou_sum / success)
    print("Saved to:", output_file)


if __name__ == "__main__":
    main()