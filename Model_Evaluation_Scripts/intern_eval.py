import argparse
import os
import json
import re
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
import ijson


def stream_json(path):
    with open(path, "r") as f:
        objects = ijson.items(f, "item")
        for obj in objects:
            yield obj


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_json", default="/home/scratch-scholars/Tawheed/Combined_1016/output.json")
    p.add_argument("--image_dir", default="/home/scratch-scholars/Tawheed/Combined_1016/")
    p.add_argument("--output_dir", default="/home/gaash/Surgical/Outputs/INTERN_VL_8B_Prompt_changed")
    p.add_argument("--model_name", default="OpenGVLab/InternVL3-8B-hf")
    p.add_argument("--load_in_4bit", action="store_true", default=True, help="Use 4-bit quantization")
    return p.parse_args()


def parse_internvl_output(text, orig_width, orig_height):
    """
    InternVL3 for grounding typically returns bounding boxes in one of:
      - [[x1, y1, x2, y2]]  normalized 0-1000
      - <box>[[x1, y1, x2, y2]]</box>
      - JSON: {"bbox_2d": [x1, y1, x2, y2], "point_2d": [cx, cy]}
    We try all formats.
    """
    think_text = ""
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    if think_match:
        think_text = think_match.group(1).strip()

    # Format 1: JSON answer tag (if model follows our prompt)
    ans_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if ans_match:
        try:
            data = json.loads(ans_match.group(1).strip())
            if isinstance(data, list):
                data = data[0]
            bbox = data.get("bbox_2d")
            point = data.get("point_2d")
            if bbox and len(bbox) == 4:
                bbox, point = scale_coords(bbox, point, orig_width, orig_height)
                return bbox, point, think_text
        except Exception:
            pass

    # Format 2: <box>[[x1, y1, x2, y2]]</box>
    box_tag = re.search(r"<box>\s*\[\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\s*\]\s*</box>", text)
    if box_tag:
        coords = [int(box_tag.group(i)) for i in range(1, 5)]
        bbox, point = scale_coords(coords, None, orig_width, orig_height)
        return bbox, point, think_text

    # Format 3: bare [[x1, y1, x2, y2]]
    bare_match = re.search(r"\[\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\s*\]", text)
    if bare_match:
        coords = [int(bare_match.group(i)) for i in range(1, 5)]
        bbox, point = scale_coords(coords, None, orig_width, orig_height)
        return bbox, point, think_text

    # Format 4: plain (x1, y1, x2, y2)
    plain_match = re.search(r"\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)", text)
    if plain_match:
        coords = [int(plain_match.group(i)) for i in range(1, 5)]
        bbox, point = scale_coords(coords, None, orig_width, orig_height)
        return bbox, point, think_text

    return None, None, think_text


def scale_coords(bbox, point, orig_width, orig_height):
    """
    InternVL uses 0-1000 normalized coordinate space.
    Convert to absolute pixel coordinates.
    """
    x1, y1, x2, y2 = bbox
    x1 = int(x1 / 1000 * orig_width)
    y1 = int(y1 / 1000 * orig_height)
    x2 = int(x2 / 1000 * orig_width)
    y2 = int(y2 / 1000 * orig_height)

    if point and len(point) == 2:
        cx = int(point[0] / 1000 * orig_width)
        cy = int(point[1] / 1000 * orig_height)
    else:
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

    print("Loading InternVL model...")
    quantization_config = BitsAndBytesConfig(load_in_4bit=True) if args.load_in_4bit else None

    processor = AutoProcessor.from_pretrained(args.model_name)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        torch_dtype=torch.float16 if not args.load_in_4bit else None,
        device_map="auto"
    )
    model.eval()
    print("Model loaded successfully")

    # InternVL3 grounding prompt — keep it direct, no lengthy format specs
    # The model natively outputs [[x1, y1, x2, y2]] in 0-1000 space
    PROMPT = (
        "Act as a senior medical surgeon performing a critical surgery. You are observing the surgical field and need to precisely locate the tool in the image to continue the procedure safely. Use the following as your reference: \n"
        "Instruction: {instruction}\n"
        "Tool name: {tool_name}\n"
        "Your response must strictly follow this format: \n"
        "{{\"bbox_2d\": [x1, y1, w, h], \"point_2d\": [cx, cy]}}\n"
        "Coordinates must be in pixel values. "
        "Focus on precision as if the patient's safety depends on it."
    )



    output_file = os.path.join(args.output_dir, "internvl_predictions.jsonl")
    total = 0
    success_count = 0

    with open(output_file, "w") as out_f:
        for item in tqdm(stream_json(args.dataset_json), desc="Processing"):
            total += 1

            try:
                img_path = os.path.join(args.image_dir, item["image"].lstrip("/"))
                if not os.path.exists(img_path):
                    print(f"Image not found: {img_path}")
                    continue

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
                            {"type": "image", "url": img_path},   # processor handles loading
                            {"type": "text", "text": prompt_text}
                        ]
                    }
                ]

                inputs = processor.apply_chat_template(
                    messages,
                    return_tensors="pt",
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                ).to(model.device, dtype=torch.float16)

                with torch.inference_mode():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=128,
                        do_sample=False
                    )

                # Trim prompt tokens from output
                output_text = processor.decode(
                    generated_ids[0, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                )

                print(f"[{item['image_id']}] Raw output: {output_text}")

                bbox, point, think_text = parse_internvl_output(output_text, orig_width, orig_height)

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
                continue

    print("\nFinished.")
    print(f"Processed : {total}")
    print(f"Success   : {success_count}")
    print(f"Success % : {(success_count / total) * 100:.2f}%" if total else "N/A")
    print(f"Saved to  : {output_file}")


if __name__ == "__main__":
    main()