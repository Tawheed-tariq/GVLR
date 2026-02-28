import argparse
import os
import json
import re
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend
import ijson
from pathlib import Path


def stream_json(path):
    with open(path, "r") as f:
        objects = ijson.items(f, "item")
        for obj in objects:
            yield obj


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_json", default="/home/scratch-scholars/Tawheed/Combined_1016/output.json")
    p.add_argument("--image_dir", default="/home/scratch-scholars/Tawheed/Combined_1016/")
    p.add_argument("--output_dir", default="/home/gaash/Surgical/Outputs/MISTRAL3_8B")
    p.add_argument("--model_name", default="mistralai/Ministral-3-8B-Instruct-2512")
    return p.parse_args()


def parse_model_output(text):
    try:
        # -------- Extract THINK --------
        think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
        think_text = think_match.group(1).strip() if think_match else ""

        # -------- Extract ANSWER --------
        ans = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
        answer_text = ans.group(1).strip() if ans else text

        # remove markdown fences
        answer_text = re.sub(r"```(?:json)?", "", answer_text)
        answer_text = answer_text.replace("```", "").strip()

        # extract JSON object or array
        json_match = re.search(r"(\{.*\}|\[.*\])", answer_text, re.DOTALL)
        if not json_match:
            return None, None, think_text

        answer_text = json_match.group(1)

        # -------- Normalize whitespace inside JSON keys --------
        answer_text = re.sub(r'[\r\n\t]+', ' ', answer_text)
        answer_text = re.sub(r'\s*"\s*:\s*', '": ', answer_text)
        answer_text = re.sub(r'"\s+([^"]+)\s+"', r'"\1"', answer_text)

        # -------- repair common JSON issues --------
        answer_text = re.sub(r"(?<![a-zA-Z])'|'(?![a-zA-Z])", '"', answer_text)
        answer_text = re.sub(r",\s*}", "}", answer_text)
        answer_text = re.sub(r",\s*]", "]", answer_text)

        try:
            data = json.loads(answer_text)
        except Exception:
            bbox_match = re.search(r'"bbox_2d"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', answer_text)
            point_match = re.search(r'"point_2d"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]', answer_text)
            if bbox_match:
                bbox = [int(bbox_match.group(i)) for i in range(1, 5)]
                if point_match:
                    point = [int(point_match.group(1)), int(point_match.group(2))]
                else:
                    point = [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]
                return bbox, point, think_text
            return None, None, think_text

        if isinstance(data, dict):
            data = [data]

        if not isinstance(data, list) or len(data) == 0:
            return None, None, think_text

        det = data[0]
        cleaned = {k.strip(): v for k, v in det.items()}

        bbox = cleaned.get("bbox_2d")
        if not bbox or len(bbox) != 4:
            return None, None, think_text

        point = cleaned.get("point_2d")
        if not point or len(point) != 2:
            point = [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]

        return bbox, point, think_text

    except Exception as e:
        print("Parse error:", e)
        return None, None, ""


def validate_bbox(bbox, w, h):
    if not bbox:
        return [0, 0, 0, 0]
    x1, y1, x2, y2 = map(int, bbox)
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return [0, 0, 0, 0]
    return [x1, y1, x2, y2]


def validate_point(pt, w, h):
    if not pt:
        return [0, 0]
    x, y = map(int, pt)
    return [max(0, min(x, w)), max(0, min(y, h))]


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading Mistral VLM...")
    tokenizer = MistralCommonBackend.from_pretrained(args.model_name)
    model = Mistral3ForConditionalGeneration.from_pretrained(
        args.model_name,
        device_map="cuda",
        torch_dtype=torch.bfloat16          
    )
    print(model.device)
    model.eval()
    print("Model loaded")

    PROMPT = """You are a precise vision localization assistant.

Return STRICT JSON:

<think>
brief reasoning
</think>
<answer>
[
{{
"bbox_2d": [x1, y1, x2, y2],
"point_2d": [cx, cy]
}}
]
</answer>

Instruction:
{instruction}
Tool name:
{tool_name}
"""

    output_file = os.path.join(args.output_dir, "mistral_predictions.jsonl")

    total = 0
    success = 0

    with open(output_file, "w") as out_f:
        for item in tqdm(stream_json(args.dataset_json), desc="Processing"):

            total += 1
            text_output = ""
            try:
                img_path = os.path.join(args.image_dir, item["image"].lstrip("/"))
                if not os.path.exists(img_path):
                    print(f"Missing image: {img_path}")
                    break
                img = Image.open(img_path).convert("RGB")
                img_data_url = Path(img_path).absolute().as_uri()

                messages = [{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": PROMPT.format(
                                instruction=item["text"],
                                tool_name=item["tool_name"]
                            )
                        },
                        # FIX 2: use "image_url" format (not "image") to match Mistral example
                        {
                            "type": "image_url",
                            "image_url": {"url": img_data_url}   # PIL Image passed directly as url value
                        },
                    ],
                }]

                # FIX 3: removed extra keyword args; match example exactly
                tokenized = tokenizer.apply_chat_template(
                    messages,
                    return_tensors="pt",
                    return_dict=True
                ).to("cuda")

                # FIX 4: move input_ids and pixel_values to device separately, matching example
                tokenized["input_ids"] = tokenized["input_ids"].to(device="cuda")
                tokenized["pixel_values"] = tokenized["pixel_values"].to(
                    dtype=torch.bfloat16, device="cuda"
                )

                image_sizes = [tokenized["pixel_values"].shape[-2:]]

                with torch.inference_mode():
                    output_ids = model.generate(
                        **tokenized,
                        image_sizes=image_sizes,
                        max_new_tokens=256,
                        do_sample=False,
                    )[0]

                # FIX 5: decode only new tokens (slice from input length), matching example
                decoded = tokenizer.decode(
                    output_ids[len(tokenized["input_ids"][0]):]
                )
                text_output = decoded.strip()
                bbox, point, think = parse_model_output(decoded)

                if bbox is None:
                    result = {
                        "image_id": item["image_id"],
                        "success": False,
                        "raw_output": decoded
                    }
                else:
                    bbox = validate_bbox(bbox, item["img_width"], item["img_height"])
                    point = validate_point(point, item["img_width"], item["img_height"])
                    success += 1

                    result = {
                        "image_id": item["image_id"],
                        "success": True,
                        "bbox": bbox,
                        "point": point,
                        "think": think,
                        "query": item["text"]
                    }

                out_f.write(json.dumps(result) + "\n")
                out_f.flush()
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error processing {item.get('image_id')}: {e}")

                result = {
                    "image_id": item.get("image_id"),
                    "success": False,
                    "raw_output": text_output
                }

                out_f.write(json.dumps(result) + "\n")
                out_f.flush()
                # FIX 6: removed `return` so errors on one item don't stop the whole run
                return

    print("\nFinished")
    print("Processed:", total)
    print("Success:", success)
    print(f"Rate: {(success / total) * 100:.2f}%")
    print("Saved to:", output_file)


if __name__ == "__main__":
    main()