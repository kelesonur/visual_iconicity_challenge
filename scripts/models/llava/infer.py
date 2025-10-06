import json
import os
import csv
import argparse
import torch
import av
import numpy as np
from tqdm import tqdm
import logging
import copy

from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from llava.conversation import conv_templates
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from pathlib import Path
import random


TASKS = [
    "handshape",
    "location",
    "path_shape",
    "path_repetition",
    "handedness",
    "transparency",
    "transparency_2",
    "reference",
    "iconicity_binary",
    "iconicity_rating",
]


# --------------------------------------------------------------------------- #
# 2. Prompt builder (exactly mirrored from infer_minimal.py)
# --------------------------------------------------------------------------- #
def build_prompt(task: str, meaning: str = "") -> str:
    if task == "iconicity_binary":
        return (
            f"Meaning: {meaning}. "
            "Some signs are iconic and some are arbitrary."
            "Find visual resemblances between the meaning and the form of the sign. "
            f"Does the sign look like or resemble '{meaning}'? Answer only one word: yes or no"
        )

    if task == "iconicity_rating":
        return (
            f"This sign means: {meaning}. "
            "Some signs are iconic and some are arbitrary."
            "Find visual resemblances between the meaning and the form of the sign. "
            f"How much does the sign look like '{meaning}'? "
            "Answer with only one number: 1,2,3,4,5,6,7 (1=not at all, 7=exactly)."
        )

    if task == "handshape":
        return (
            "Major sign handshape? Answer with only one: H1,H2,H3,H4,H5,H6,H7"
            "(H1=all fingers closed to a fist, H2=all fingers extended, H3=all fingers curved or clawed, H4=one (selected) finger extended, H5=one (selected) finger curved or clawed, H6= two or more (selected) fingers extended, H7=two or more(selected) fingers curved or clawed)."
        )

    if task == "location":
        return (
            "Major sign location? Answer with only one: L1, L2, L3, L4, L5"
            "(L1=hands touching head/face, L2=hands touching torso, L3=hands touching arm, L4=hands touching weak/passive hand, L5=hands in front of the body or face)"
        )

    if task == "path_shape":
        return (
            "Movement path shape? Answer with only one: Hold, Straight, Arched, Circular."
            "(Hold=no path or direction, Straight=move in a straight line, Arched=move in an arched line, Circular=move in a circular path)"
        )

    if task == "path_repetition":
        return (
            "Movement repetition? Answer with only one: Single, Repeated."
            "(Single=one movement, Repeated=multiple or repeated movements)"
        )

    if task == "handedness":
        return (
            "Handedness? Answer with only one: One-handed, Two-handed symmetrical, Two-handed asymmetrical."
            "(One-handed=only one hand is used in the sign, Two-handed symmetrical=two hands are used but the hands move together and have the same handshape, Two-handed asymmetrical=two hands are visible, but one hand does not move and the hands have different handshapes)"
        )

    if task == "reference":
        return (    
            "Reference? Answer with only one: Object, Action, Other."
            "(Object=the sign resembles an object, Action=the sign resembles an action, Other=the sign does not resemble any of the above)"
        )

    if task == "transparency":
        all_glosses = [
            "AMBULANCE", "BABY", "BALL", "BEAR", "BED", "BICYCLE", "BIRD", "BLANKET", "BOTTLE", "BOX",
            "BRIDGE", "BUILDING", "BUS", "BUTTERFLY", "CALCULATOR", "CAMEL", "CAR", "CAT", "CELL", "CHICKEN",
            "CURTAINS", "DEER", "DOCTOR", "DOG", "DOOR", "DRILL", "EAGLE", "ELECTRICITY", "ELEPHANT", "FIRE",
            "FRUIT", "HELICOPTER", "HORSE", "INTERPRETER", "KEY", "KIWI", "LAPTOP", "LIGHTBULB", "LOBSTER", "MONKEY",
            "MUMMY", "PENGUIN", "PERSON", "PIANO", "PISTOL", "PLANE", "PRAM", "PUPPET", "RABBIT", "RATTLE",
            "RESTAURANT", "ROOM", "SHEEP", "SNAKE", "SOFA", "SPIDER", "SPOON", "SUGAR", "SUITCASE", "TABLE",
            "TELEPHONE", "TO-ARGUE", "TO-BREAK", "TO-COOK", "TO-CRASH", "TO-CRY", "TO-CUT", "TO-DIE", "TO-ERASE", "TO-FLY",
            "TO-GO-OUT", "TO-GOSSIP", "TOILET", "TO-INJECT", "TO-JUGGLE", "TO-KNOCK", "TO-LAUGH", "TO-ORDER", "TO-PLAY-CARDS", "TO-PUMP",
            "TO-PUT-CLOTHES-ON", "TO-SHOUT", "TO-SLAP", "TO-SMS", "TO-STAPLE", "TO-STEAL", "TO-SWIM", "TOWEL", "TO-WRING", "TREE",
            "UMBRELLA", "WATER", "WHEELCHAIR", "WINDSCREEN WIPER", "ZIMMER"
        ]
        
        gloss_options = ", ".join(all_glosses)
        return (
            "What does this sign resemble? Look at the form and movement of the sign. "
            f"Choose the most likely option from these possibilities: {gloss_options}. "
            "Answer with only the exact word from the list that best matches what the sign looks like."
        )

    return f"Analyze the sign's {task.replace('_', ' ')}."


# --------------------------------------------------------------------------- #
# 3. I/O utilities
# --------------------------------------------------------------------------- #
def load_json(path: Path):
    with path.open() as fh:
        return json.load(fh)

def save_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["gloss", "prediction"])
        # Extract just gloss and prediction from the full row data
        for row in rows:
            vid, task, gloss, dutch, category, pred = row
            wr.writerow([gloss, pred])


def save_unified_csv(all_results, path: Path):
    """Save all tasks results to a single CSV file"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        # Headers: video, gloss, dutch, category, then one column per task
        headers = ["video", "gloss", "dutch", "category"] + TASKS
        wr.writerow(headers)
        
        # Group results by video
        video_data = {}
        for row in all_results:
            video, task, gloss, dutch, category, prediction = row
            if video not in video_data:
                video_data[video] = {
                    "gloss": gloss,
                    "dutch": dutch,
                    "category": category,
                    "predictions": {}
                }
            video_data[video]["predictions"][task] = prediction
        
        # Write one row per video with all task predictions
        for video, data in video_data.items():
            row = [video, data["gloss"], data["dutch"], data["category"]]
            for task in TASKS:
                row.append(data["predictions"].get(task, ""))
            wr.writerow(row)


def setup_logger(path: Path):
    """Sets up a logger to file and console, creating the directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # To prevent duplicate handlers if called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()
        
    fh = logging.FileHandler(path)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s — %(levelname)s — %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# --------------------------------------------------------------------------- #
# 4. Inference
# --------------------------------------------------------------------------- #
def run_all_tasks(args):
    """Run inference for all tasks and save to a single unified CSV"""
    # CUDA environment is already set at module level
    
    root = Path(args.base_data_path)
    video_dir = root / "videos"
    
    # Extract model name for unique filenames
    model_path = args.base_model_path
    if "LLaVA-Video-7B-Qwen2" in model_path:
        model_name = "LLaVA-Video-7B-Qwen2"
    elif "LLaVA-Video-72B-Qwen2" in model_path:
        model_name = "LLaVA-Video-72B-Qwen2"
    elif "llava-onevision-qwen2-7b-ov" in model_path:
        model_name = "LLaVA-OneVision-7B"
    elif "LLaVA-NeXT" in model_path:
        model_name = "LLaVA-NeXT-7B"
    else:
        model_name = Path(model_path).name
    results = Path(args.results_path) / f"Iconicity_all_tasks_preds_{model_name}.csv"
    log_path = Path(args.results_path) / f"log_{model_name}_all_tasks.txt"
    
    logger = setup_logger(log_path)
    
    # Load unified mappings and build index/list
    video_mapping_path = root / "video_mappings.json"
    if not video_mapping_path.exists():
        raise FileNotFoundError(f"Missing master JSON: {video_mapping_path}")
    _mapping_items = load_json(video_mapping_path)
    if isinstance(_mapping_items, list):
        mapping_items = _mapping_items
        video_mapping = {it.get("video", ""): it for it in _mapping_items}
    elif isinstance(_mapping_items, dict):
        video_mapping = _mapping_items
        mapping_items = list(_mapping_items.values())
    else:
        video_mapping = {}
        mapping_items = []
    logger.info(f"Loaded video mapping with {len(mapping_items)} entries")

    # Load model
    model_kwargs = {"torch_dtype": "bfloat16", "device_map": "auto", "attn_implementation": "sdpa"}
    if args.load_4bit:
        model_kwargs["load_4bit"] = True
    if args.load_8bit:
        model_kwargs["load_8bit"] = True
    
    tokenizer, model, processor, max_length = load_pretrained_model(
        args.base_model_path, 
        None, 
        "llava_qwen", 
        **model_kwargs
    )
    model.eval()
    logger.info("Model loaded successfully")

    all_results = []
    
    # Process each video for all tasks
    for item in tqdm(mapping_items, desc="Processing videos for all tasks"):
        vid = item["video"]
        
        # Extract metadata once per video
        gloss = item.get("gloss", "")
        dutch = item.get("dutch", "")
        category = item.get("category", "")
        
        modal_path = os.path.join(video_dir, vid)
        video_frames = load_video(modal_path)
        if video_frames is None:
            logger.warning(f"Skipping corrupted or unreadable video: {vid}")
            continue

        target_dtype = getattr(model, "dtype", torch.float16)
        if target_dtype not in (torch.float16, torch.bfloat16):
            target_dtype = torch.float16
        # Keep on CPU; Accelerate will route to the correct GPU shard for the vision tower
        video_frames = processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].to(dtype=target_dtype)
        video_frames = [video_frames]

        # Run inference for each task
        for task in TASKS:
            if task == "transparency_2":
                # Build 10-option choices from all glosses in mapping
                all_glosses = [it.get("gloss", "") for it in mapping_items if it.get("gloss", "")]
                distractors = [g for g in all_glosses if g and g != gloss]
                num_distractors = min(9, len(distractors))
                sampled = random.sample(distractors, num_distractors) if num_distractors > 0 else []
                options = [gloss] + sampled
                random.shuffle(options)
                options_str = ", ".join(options)
                prompt = (
                    "What does this sign resemble? Look at the form and movement of the sign. "
                    f"Choose the most likely option from these possibilities: {options_str}. "
                    "Answer with only the exact word from the list that best matches what the sign looks like."
                )
            else:
                prompt = build_prompt(task, gloss)
            
            conv_template = "qwen_1_5"
            question = DEFAULT_IMAGE_TOKEN + f"{prompt}"
            conv = copy.deepcopy(conv_templates[conv_template])
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()
            if task == "transparency_2":
                logger.info(f"{vid} | transparency_2 prompt: {prompt_question}")

            with torch.no_grad():
                input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
                cont = model.generate(
                    input_ids,
                    images=video_frames,
                    modalities=["video"],
                    do_sample=False,
                    temperature=0.0,
                    max_new_tokens=128,
                )
            pred = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
            logger.info(f"{vid} | {task}: {pred}")
            all_results.append((vid, task, gloss, dutch, category, pred))

    save_unified_csv(all_results, results)
    logger.info(f"Saved unified results → {results}")


def load_video(video_path):
    """
    Safely loads a video, returning None if it fails.
    If the path points to an .mpeg file, it tries to load the corresponding .mp4 version from the original file's location.
    """
    # Resolve the symlink to get the real path of the video
    try:
        real_video_path = os.path.realpath(video_path)
    except OSError:
        # Fallback to using the original path if resolving fails
        real_video_path = str(video_path)

    path_to_load = real_video_path
    
    # If the original file is an .mpeg, check for an .mp4 version in the same directory
    if real_video_path.lower().endswith('.mpeg'):
        mp4_path = os.path.splitext(real_video_path)[0] + '.mp4'
        if os.path.exists(mp4_path):
            path_to_load = mp4_path
            
    try:
        container = av.open(str(path_to_load))
        if not container.streams.video:
            return None
        
        total_frames = container.streams.video[0].frames
        if total_frames == 0:
            return None

        indices = np.linspace(0, total_frames - 1, 8, dtype=int)

        frames = []
        container.seek(0)
        # Efficiently seek to the first frame and decode only what's needed
        for i, frame in enumerate(container.decode(video=0)):
            if i > indices[-1]:
                break
            if i in indices:
                frames.append(frame.to_ndarray(format="rgb24"))
        
        if not frames:
            return None
            
        return np.stack(frames)
    except Exception as e:
        print(f"Warning: Failed to load video {path_to_load} (original path: {video_path}) due to: {e}")
        return None


def run(args):
    """Run inference for a single task using unified video_mappings.json"""
    # CUDA environment is already set at module level
    
    root      = Path(args.base_data_path)
    video_dir = root / "videos"
    
    # Extract model name for unique filenames
    model_path = args.base_model_path
    if "LLaVA-Video-7B-Qwen2" in model_path:
        model_name = "LLaVA-Video-7B-Qwen2"
    elif "LLaVA-Video-72B-Qwen2" in model_path:
        model_name = "LLaVA-Video-72B-Qwen2"
    elif "llava-onevision-qwen2-7b-ov" in model_path:
        model_name = "LLaVA-OneVision-7B"
    elif "LLaVA-NeXT" in model_path:
        model_name = "LLaVA-NeXT-7B"
    else:
        model_name = Path(model_path).name
    results   = Path(args.results_path) / f"Iconicity_{args.task}_preds_{model_name}.csv"
    log_path  = Path(args.results_path) / f"log_{model_name}_{args.task}.txt"
    
    logger = setup_logger(log_path)
    
    # Load unified mappings and build index/list
    video_mapping_path = root / "video_mappings.json"
    if not video_mapping_path.exists():
        raise FileNotFoundError(f"Missing master JSON: {video_mapping_path}")
    _mapping_items = load_json(video_mapping_path)
    if isinstance(_mapping_items, list):
        mapping_items = _mapping_items
    elif isinstance(_mapping_items, dict):
        mapping_items = list(_mapping_items.values())
    else:
        mapping_items = []
    logger.info(f"Loaded {len(mapping_items)} items from {video_mapping_path}")

    # Load model
    model_kwargs = {"torch_dtype": "bfloat16", "device_map": "auto", "attn_implementation": "sdpa"}
    if args.load_4bit:
        model_kwargs["load_4bit"] = True
    if args.load_8bit:
        model_kwargs["load_8bit"] = True
    
    tokenizer, model, processor, max_length = load_pretrained_model(
        args.base_model_path, 
        None, 
        "llava_qwen", 
        **model_kwargs
    )
    model.eval()
    logger.info("Model loaded successfully")

    rows = []
    for item in tqdm(mapping_items, desc=f"Predicting {args.task}"):
        vid  = item["video"]
        gloss = item.get("gloss", "")
        dutch = item.get("dutch", "")
        category = item.get("category", "")
        
        if args.task == "transparency_2":
            all_glosses = [it.get("gloss", "") for it in mapping_items if it.get("gloss", "")]
            distractors = [g for g in all_glosses if g and g != gloss]
            num_distractors = min(9, len(distractors))
            sampled = random.sample(distractors, num_distractors) if num_distractors > 0 else []
            options = [gloss] + sampled
            random.shuffle(options)
            options_str = ", ".join(options)
            prompt = (
                "What does this sign resemble? Look at the form and movement of the sign. "
                f"Choose the most likely option from these possibilities: {options_str}. "
                "Answer with only the exact word from the list that best matches what the sign looks like."
            )
        else:
            prompt  = build_prompt(args.task, gloss)
        
        modal_path = os.path.join(video_dir, vid)
        video_frames = load_video(modal_path)
        if video_frames is None:
            logger.warning(f"Skipping corrupted or unreadable video: {vid}")
            continue

        target_dtype = getattr(model, "dtype", torch.float16)
        if target_dtype not in (torch.float16, torch.bfloat16):
            target_dtype = torch.float16
        # Keep on CPU; Accelerate will route to the correct GPU shard for the vision tower
        video_frames = processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].to(dtype=target_dtype)
        video_frames = [video_frames]

        conv_template = "qwen_1_5"
        question = DEFAULT_IMAGE_TOKEN + f"{prompt}"
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        if args.task == "transparency_2":
            logger.info(f"{vid} | transparency_2 prompt: {prompt_question}")

        with torch.no_grad():
            input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
            cont = model.generate(
                input_ids,
                images=video_frames,
                modalities=["video"],
                do_sample=False,
                temperature=0.0,
                max_new_tokens=128,
            )
        pred = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
        logger.info(f"{vid} | {pred}")
        if args.task == "transparency_2":
            logger.info(f"{vid} | transparency_2 answer: {pred}")
        
        rows.append((vid, args.task, gloss, dutch, category, pred))

    save_csv(rows, results)
    logger.info(f"Saved → {results}")


# --------------------------------------------------------------------------- #
# 5. CLI
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLaVA-NeXT inference for sign language iconicity tasks")
    parser.add_argument("--base_model_path", required=True, help="Path to LLaVA-NeXT model")
    parser.add_argument("--base_data_path", required=True, help="Path to data directory")
    parser.add_argument("--task", choices=TASKS, help="Single task to run (for single-task mode)")
    parser.add_argument("--all_tasks", action="store_true", help="Run all tasks and create unified CSV")
    parser.add_argument("--results_path", default="results", help="Output directory for results")
    parser.add_argument("--device_ids", default="0", help="CUDA device IDs")
    parser.add_argument("--load_4bit", action="store_true", help="Use 4-bit quantization to save memory")
    parser.add_argument("--load_8bit", action="store_true", help="Use 8-bit quantization to save memory")
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_ids
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" 
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    if args.all_tasks:
        run_all_tasks(args)
    elif args.task:
        run(args)
    else:
        parser.error("Must specify either --task or --all_tasks")
