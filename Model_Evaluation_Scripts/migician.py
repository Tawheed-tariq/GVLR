import argparse
import os
import json
import re
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import ijson


def stream_json(path):
    with open(path, "r") as f:
        objects = ijson.items(f, "item")
        for obj in objects:
            yield obj


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_json", default="/home/scratch-btech/Tawheed/SurgicalDataset/Combined_1016/output.json")
    p.add_argument("--image_dir", default="/home/scratch-btech/Tawheed/SurgicalDataset/Combined_1016/")
    p.add_argument("--output_dir", default="/home/2022bite008/Surgical/Outputs/Migician")
    p.add_argument("--model_name", default="Michael4933/Migician")
    p.add_argument("--max_image_size", type=int, default=1024)
    return p.parse_args()


def resize(img: Image.Image, max_size=1024) -> Image.Image:
    """Resize image to fit within max_size while keeping aspect ratio."""
    w, h = img.size
    scale = min(max_size / max(w, h), 1.0)
    if scale < 1:
        img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
    return img


def parse_migician_output(text, orig_width, orig_height):
    """
    Migician outputs bounding boxes in the format:
      (x1, y1, x2, y2) normalized to [0, 1000]
    e.g. <box>(123,456,789,012)</box>  or  [[x1,y1,x2,y2]]
    We try both formats.
    """
    think_text = ""
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    if think_match:
        think_text = think_match.group(1).strip()

    # Format 1: <box>(x1,y1,x2,y2)</box>
    box_match = re.search(r"<box>\s*\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)\s*</box>", text)
    if box_match:
        coords = [int(box_match.group(i)) for i in range(1, 5)]
        bbox, point = normalize_to_pixel(coords, orig_width, orig_height)
        return bbox, point, think_text

    # Format 2: [[x1, y1, x2, y2]]  (sometimes Migician returns JSON-style)
    json_match = re.search(r"\[\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\s*\]", text)
    if json_match:
        coords = [int(json_match.group(i)) for i in range(1, 5)]
        bbox, point = normalize_to_pixel(coords, orig_width, orig_height)
        return bbox, point, think_text

    # Format 3: plain (x1,y1,x2,y2) anywhere in the text
    plain_match = re.search(r"\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)", text)
    if plain_match:
        coords = [int(plain_match.group(i)) for i in range(1, 5)]
        bbox, point = normalize_to_pixel(coords, orig_width, orig_height)
        return bbox, point, think_text

    return None, None, think_text


def normalize_to_pixel(coords, orig_width, orig_height):
    """
    Migician uses a 0–1000 coordinate space.
    Convert to absolute pixel coordinates.
    """
    x1, y1, x2, y2 = coords
    x1 = int(x1 / 1000 * orig_width)
    y1 = int(y1 / 1000 * orig_height)
    x2 = int(x2 / 1000 * orig_width)
    y2 = int(y2 / 1000 * orig_height)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    return [x1, y1, x2, y2], [cx, cy]


def validate_bbox(bbox, max_w, max_h):
    if not bbox or len(bbox) != 4:
        return [0, 0, 0, 0]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), max_w))
    y1 = max(0, min(int(y1), max_h))
    x2 = max(0, min(int(x2), max_w))
    y2 = max(0, min(int(y2), max_h))
    if x2 <= x1 or y2 <= y1:
        return [0, 0, 0, 0]
    return [x1, y1, x2, y2]


def validate_point(point, max_w, max_h):
    if not point or len(point) != 2:
        return [0, 0]
    x, y = point
    x = max(0, min(int(x), max_w))
    y = max(0, min(int(y), max_h))
    return [x, y]


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading Migician model...")
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    print("Model loaded successfully")

    # Migician uses Qwen2-VL's grounding token syntax
    PROMPT = (
        "Locate the <|object_ref_start|>{tool_name}<|object_ref_end|> in the image. "
        "{instruction}. "
        "Output only a JSON object like this, no explanation:\n"
        "{{\"bbox_2d\": [x1, y1, x2, y2], \"point_2d\": [cx, cy]}}"
    )

    output_file = os.path.join(args.output_dir, "migician_predictions.jsonl")
    total = 0
    success_count = 0

    with open(output_file, "w") as out_f:
        for item in tqdm(stream_json(args.dataset_json), desc="Processing"):
            total += 1

            try:
                img_path = os.path.join(args.image_dir, item["image"].lstrip("/"))
                if not os.path.exists(img_path):
                    print(f"Image not found: {img_path}")
                    return                          # skip missing images instead of aborting

                img = Image.open(img_path).convert("RGB")
                img = resize(img, max_size=args.max_image_size)

                orig_width = item["img_width"]
                orig_height = item["img_height"]

                prompt_text = PROMPT.format(
                    instruction=item["text"],
                    tool_name=item["tool_name"]
                )

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": prompt_text}
                        ]
                    }
                ]

                text_input = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text_input],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt"
                ).to(model.device)

                with torch.inference_mode():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=128,
                        do_sample=False
                    )

                trimmed = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]

                print(f"[{item['image_id']}] Raw output: {output_text}")

                bbox, point, think_text = parse_migician_output(output_text, orig_width, orig_height)

                if bbox is None:
                    result = {
                        "image_id": item["image_id"],
                        "success": False,
                        "raw_output": output_text
                    }
                else:
                    bbox_valid = validate_bbox(bbox, orig_width, orig_height)
                    point_valid = validate_point(point, orig_width, orig_height)
                    success_count += 1
                    result = {
                        "image_id": item["image_id"],
                        "success": True,
                        "bbox": bbox_valid,
                        "point": point_valid,
                        "think": think_text,
                        "query": item["text"]
                    }

                out_f.write(json.dumps(result) + "\n")
                out_f.flush()

                del inputs, generated_ids
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error processing {item.get('image_id')}: {e}")
                return                              # skip errored items, don't abort

    print("\nFinished.")
    print(f"Processed : {total}")
    print(f"Success   : {success_count}")
    print(f"Success % : {(success_count / total) * 100:.2f}%" if total else "N/A")
    print(f"Saved to  : {output_file}")


if __name__ == "__main__":
    main()