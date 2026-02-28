import argparse
import os
import json
import re
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
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
    p.add_argument("--output_dir", default="/home/gaash/Surgical/Outputs/QWEN_2_5_Prompt_changed")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    return p.parse_args()


def parse_qwen_output(text):
    try:
        ans = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
        if ans is None:
            return None, None, ""

        answer_text = ans.group(1).strip()

        # Try strict JSON first
        try:
            data = json.loads(answer_text)
        except json.JSONDecodeError:
            # Attempt light repair
            answer_text = answer_text.replace("'", '"')
            answer_text = re.sub(r",\s*}", "}", answer_text)
            answer_text = re.sub(r",\s*]", "]", answer_text)
            try:
                data = json.loads(answer_text)
            except:
                return None, None, ""

        if isinstance(data, dict):
            data = [data]

        if not isinstance(data, list) or len(data) == 0:
            return None, None, ""

        detection = data[0]

        cleaned_detection = {}
        for k, v in detection.items():
            cleaned_key = k.strip()
            cleaned_detection[cleaned_key] = v

        bbox = cleaned_detection.get("bbox_2d")
        if not bbox or len(bbox) != 4:
            return None, None, ""

        point = cleaned_detection.get("point_2d")
        if not point or len(point) != 2:
            point = [
                (bbox[0] + bbox[2]) // 2,
                (bbox[1] + bbox[3]) // 2
            ]

        think = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
        think_text = think.group(1).strip() if think else ""

        return bbox, point, think_text

    except Exception as e:
        print("Parse error:", e)
        return None, None, ""

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

    print("Loading Qwen3 model...")
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    print("Model loaded successfully")

    # PROMPT = """
    #     You are a precise vision localization assistant.

    #     Return STRICT JSON format:

    #     <think>
    #     brief reasoning
    #     </think>
    #     <answer>
    #     [
    #     {{
    #     "bbox_2d": [x1, y1, x2, y2],
    #     "point_2d": [cx, cy]
    #     }}
    #     ]
    #     </answer>

    #     Instruction:
    #     {instruction}
    #     Tool name:
    #     {tool_name}
    # """

    PROMPT = """
    Act as a senior medical surgeon performing a critical surgery. You are observing the surgical field and need to precisely locate the tool in the image to continue the procedure safely. Use the following as your reference:  

    Instruction: {instruction}  
    Tool name: {tool_name}  

    Your response must strictly follow this format:  

    <think>
    Provide brief reasoning for your identification. Keep it concise and clinically relevant.
    </think>

    <answer>
    [
    {{
        "bbox_2d": [x, y, w, h],   
        "point_2d": [cx, cy]           
    }}
    ]
    </answer>

    Rules:
    1. Do not include any text outside <think> and <answer> tags.  
    2. Always return a valid JSON object even if uncertain.  
    3. Coordinates must be in pixel values.  
    4. Focus on precision as if the patient's safety depends on it.
    """

    output_file = os.path.join(args.output_dir, "qwen3_predictions.jsonl")

    total = 0
    success_count = 0

    with open(output_file, "w") as out_f:
        for item in tqdm(stream_json(args.dataset_json), desc="Processing"):

            total += 1

            try:
                img_path = os.path.join(args.image_dir, item["image"].lstrip("/"))
                if not os.path.exists(img_path):
                    print(f"Image: {img_path} does not exist")
                    return

                img = Image.open(img_path).convert("RGB")
                orig_width = item["img_width"]
                orig_height = item["img_height"]

                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": PROMPT.format(instruction=item["text"], tool_name=item["tool_name"])}
                    ]
                }]

                print(messages)

                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt"
                ).to(model.device)

                with torch.inference_mode():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=False
                    )

                trimmed = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]

                output_text = processor.batch_decode(
                    trimmed,
                    skip_special_tokens=True
                )[0]

                print(output_text)

                bbox, point, think_text = parse_qwen_output(output_text)

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
                return

    print("\nFinished.")
    print(f"Processed: {total}")
    print(f"Success: {success_count}")
    print(f"Success rate: {(success_count/total)*100:.2f}%")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
