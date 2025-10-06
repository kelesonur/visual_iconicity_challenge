#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen2.5-VL few-shot inference for seven tasks for benchmarking iconicity.
Uses random 4-shot examples: 2 arbitrary, 1 high iconic, 1 low iconic.
"""

import os
import sys
import csv
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import random

import torch
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None  # type: ignore
try:
    from transformers import Qwen2VLForConditionalGeneration
except Exception:  # older/newer transformers without Qwen2-VL
    Qwen2VLForConditionalGeneration = None  # type: ignore

# Use the bundled utils to avoid external network usage
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    sys.path.append(str(Path(__file__).parent))
    from qwen_vl_utils import process_vision_info


# ----------------------------- Tasks ----------------------------- #
TASKS = [
    "handshape",
    "location",
    "path_shape",
    "path_repetition",
    "handedness",
    "transparency",
    "transparency_2",
    "iconicity_rating",
]

# Few-shot examples will be selected randomly at runtime
FEWSHOT_EXAMPLES = []
FEWSHOT_VIDEOS = set()


# ----------------------------- I/O utilities -------------------------------- #
def load_json(path: Path):
    with path.open() as fh:
        return json.load(fh)


def load_gold_labels(csv_path: Path) -> Dict[str, Dict[str, str]]:
    """Load gold labels from CSV file."""
    gold_labels = {}
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            video = row["video"]
            gold_labels[video] = row
    return gold_labels


def select_random_fewshot_examples(gold_labels: Dict[str, Dict[str, str]], seed: int = None) -> List[Dict[str, str]]:
    """Randomly select 4-shot examples: 2 arbitrary, 1 high iconic, 1 low iconic."""
    if seed is not None:
        random.seed(seed)
    
    # Group videos by category
    categories = {"arbitrary": [], "high iconic": [], "low iconic": []}
    
    for video, labels in gold_labels.items():
        category = labels.get("category", "").strip()
        if category in categories:
            # Convert CSV row to example format
            example = {
                "video": video,
                "gloss": labels.get("gloss", ""),
                "category": category,
                "handshape": labels.get("handshape", ""),
                "location": labels.get("location", ""),
                "path_shape": labels.get("path_shape", ""),
                "path_repetition": labels.get("path_repetition", ""),
                "handedness": labels.get("handedness", ""),
                "iconicity_rating": labels.get("iconicity_rating", ""),
                "transparency": labels.get("gloss", "")  # Use gloss as transparency answer
            }
            categories[category].append(example)
    
    # Select examples according to the specified distribution
    selected_examples = []
    
    # Select 2 arbitrary examples
    if len(categories["arbitrary"]) >= 2:
        arbitrary_selected = random.sample(categories["arbitrary"], 2)
        selected_examples.extend(arbitrary_selected)
        for ex in arbitrary_selected:
            print(f"Selected arbitrary example: {ex['video']} - {ex['gloss']}")
    else:
        print(f"Warning: Not enough arbitrary examples (found {len(categories['arbitrary'])}, need 2)")
    
    # Select 1 high iconic example
    if len(categories["high iconic"]) >= 1:
        high_iconic_selected = random.sample(categories["high iconic"], 1)
        selected_examples.extend(high_iconic_selected)
        for ex in high_iconic_selected:
            print(f"Selected high iconic example: {ex['video']} - {ex['gloss']}")
    else:
        print(f"Warning: Not enough high iconic examples (found {len(categories['high iconic'])}, need 1)")
    
    # Select 1 low iconic example
    if len(categories["low iconic"]) >= 1:
        low_iconic_selected = random.sample(categories["low iconic"], 1)
        selected_examples.extend(low_iconic_selected)
        for ex in low_iconic_selected:
            print(f"Selected low iconic example: {ex['video']} - {ex['gloss']}")
    else:
        print(f"Warning: Not enough low iconic examples (found {len(categories['low iconic'])}, need 1)")
    
    return selected_examples


def save_unified_csv(all_results: List[tuple], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        headers = ["video", "gloss", "dutch", "category"] + TASKS
        wr.writerow(headers)

        # Aggregate by video
        by_vid = {}
        for (video, gloss, dutch, category, task, pred) in all_results:
            if video not in by_vid:
                by_vid[video] = {"gloss": gloss, "dutch": dutch, "category": category, "pred": {}}
            by_vid[video]["pred"][task] = pred

        for video, data in by_vid.items():
            row = [video, data["gloss"], data.get("dutch", ""), data.get("category", "")]
            for t in TASKS:
                row.append(data["pred"].get(t, ""))
            wr.writerow(row)


def save_task_csv(rows: List[tuple], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["video", "task", "gloss", "prediction"])
        wr.writerows(rows)


# ------------------------------- Inference ---------------------------------- #
def build_task_question(task: str, gloss: str = "") -> str:
    """Build the question for a specific task."""
    if task == "iconicity_rating":
        return f"This sign means: {gloss}. Some signs are iconic and some are arbitrary. Find visual resemblances between the meaning and the form of the sign. How much does the sign look like '{gloss}'? Answer with only one number: 1,2,3,4,5,6,7 (1=not at all, 7=exactly)."
    elif task == "handshape":
        return "Major sign handshape? Answer with only one: H1,H2,H3,H4,H5,H6,H7 (H1=all fingers closed to a fist, H2=all fingers extended, H3=all fingers curved or clawed, H4=one (selected) finger extended, H5=one (selected) finger curved or clawed, H6= two or more (selected) fingers extended, H7=two or more(selected) fingers curved or clawed)."
    elif task == "location":
        return "Major sign location? Answer with only one: L1, L2, L3, L4, L5 (L1=hands touching head/face, L2=hands touching torso, L3=hands touching arm, L4=hands touching weak/passive hand, L5=hands in front of the body or face)"
    elif task == "path_shape":
        return "Movement path shape? Answer with only one: Hold, Straight, Arched, Circular. (Hold=no path or direction, Straight=move in a straight line, Arched=move in an arched line, Circular=move in a circular path)"
    elif task == "path_repetition":
        return "Movement repetition? Answer with only one: Single, Repeated. (Single=one movement, Repeated=multiple or repeated movements)"
    elif task == "handedness":
        return "Handedness? Answer with only one: One-handed, Two-handed symmetrical, Two-handed asymmetrical. (One-handed=only one hand is used in the sign, Two-handed symmetrical=two hands are used but the hands move together and have the same handshape, Two-handed asymmetrical=two hands are visible, but one hand does not move and the hands have different handshapes)"
    elif task == "transparency":
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
        return f"What does this sign resemble? Look at the form and movement of the sign. Choose the most likely option from these possibilities: {gloss_options}. Answer with only the exact word from the list that best matches what the sign looks like. If the sign does not resemble any of the above, answer 'UNKNOWN'"
    else:
        return f"Analyze the sign's {task.replace('_', ' ')}."


def get_task_answer(task: str, example: Dict[str, str]) -> str:
    """Get the correct answer for a task from the example."""
    if task == "iconicity_rating":
        # Round iconicity rating to nearest integer
        rating = float(example["iconicity_rating"])
        return str(int(round(rating)))
    elif task == "handshape":
        return example["handshape"]
    elif task == "location":
        return example["location"]
    elif task == "path_shape":
        return example["path_shape"]
    elif task == "path_repetition":
        return example["path_repetition"]
    elif task == "handedness":
        return example["handedness"]
    elif task == "transparency":
        return example["transparency"]
    else:
        return "unknown"


def build_messages_for_video_with_fewshot(video_path: str, task: str, test_gloss: str, fewshot_examples: List[Dict], video_dir: Path):
    """Build proper few-shot conversation with clear example-answer pairs for Qwen."""
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that analyzes sign language videos."
        }
    ]
    
    # Add each few-shot example as a separate user-assistant exchange
    for i, example in enumerate(fewshot_examples):
        fewshot_video = example["video"]
        fewshot_path = str(video_dir / fewshot_video)
        
        # Handle both .mp4 and .mpeg extensions
        if not Path(fewshot_path).exists():
            if fewshot_path.endswith('.mp4'):
                alt_path = fewshot_path.replace('.mp4', '.mpeg')
            elif fewshot_path.endswith('.mpeg'):
                alt_path = fewshot_path.replace('.mpeg', '.mp4')
            else:
                alt_path = None
            
            if alt_path and Path(alt_path).exists():
                fewshot_path = alt_path
        
        # Build question for this task
        question = build_task_question(task, example["gloss"])
        
        # Get correct answer for this example
        answer = get_task_answer(task, example)
        
        # User message with video and question
        messages.append({
            "role": "user",
            "content": [
                {"type": "video", "video": fewshot_path},
                {"type": "text", "text": question}
            ]
        })
        
        # Assistant response with correct answer
        messages.append({
            "role": "assistant",
            "content": answer
        })
    
    # Now add the test video
    test_question = build_task_question(task, test_gloss)
    
    # Final user message with test video and question
    messages.append({
        "role": "user",
        "content": [
            {"type": "video", "video": video_path},
            {"type": "text", "text": test_question}
        ]
    })
    
    return messages


def extract_frames_with_pyav(video_path: Path, max_frames_num: int = 4) -> Tuple[List[str], List[float]]:
    import av
    import numpy as np
    from PIL import Image

    container = av.open(str(video_path))
    total_frames = container.streams.video[0].frames
    indices = np.linspace(0, total_frames - 1, max_frames_num).astype(int)
    video_fps = container.streams.video[0].average_rate
    frame_time = [int(i / video_fps) if video_fps else 0 for i in indices]

    def read_video_pyav(container, indices):
        frames = []
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)
        return [x.to_ndarray(format="rgb24") for x in frames]

    clip = read_video_pyav(container, indices)

    frame_paths = []
    out_dir = Path("./image")
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(clip):
        img = Image.fromarray(frame)
        fp = out_dir / f"frame{i+1}.jpg"
        img.save(fp)
        frame_paths.append(str(fp))
    return frame_paths, frame_time


def _find_cached_processor_dir(repo_id: str) -> Path:
    """Return a cached snapshot directory that contains preprocessor_config.json for the given repo_id.
    Example repo_id: 'Qwen/Qwen2.5-VL-7B-Instruct'
    """
    cache_root = os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
    hub_dir = Path(cache_root) / "hub" / ("models--" + repo_id.replace("/", "--")) / "snapshots"
    if not hub_dir.exists():
        return Path("")
    # Prefer latest snapshot
    for snap in sorted(hub_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if (snap / "preprocessor_config.json").exists():
            return snap
    return Path("")


def _load_processor_with_fallback(base_model_path: str) -> AutoProcessor:
    """Try loading processor from the provided path/ID; fallback to cached repo if needed (offline friendly)."""
    # 1) Try direct (path or repo id)
    try:
        return AutoProcessor.from_pretrained(base_model_path)
    except Exception:
        pass

    # 2) Try local-only from canonical repo id (most common cache key)
    candidates = []
    base_str = str(base_model_path)
    if "Qwen2-VL" in base_str:
        candidates.append("Qwen/Qwen2-VL-7B")
    if "Qwen2.5-VL" in base_str or not candidates:
        # default to 2.5 instruct
        candidates.append("Qwen/Qwen2.5-VL-7B-Instruct")

    for cand in candidates:
        try:
            return AutoProcessor.from_pretrained(cand, local_files_only=True)
        except Exception:
            continue

    # 3) Scan cache snapshots for a processor
    scan_repo = candidates[0]
    snap = _find_cached_processor_dir(scan_repo)
    if snap and snap.exists():
        return AutoProcessor.from_pretrained(str(snap))

    # 4) Give a clear error
    raise RuntimeError(
        "Could not load AutoProcessor offline. Either point --base_model_path to a folder containing preprocessor_config.json "
        "or temporarily enable downloads to fetch processor files (unset HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE)."
    )


def _select_model_class(base_model_path: str):
    base_str = str(base_model_path)
    if "Qwen2-VL" in base_str and Qwen2VLForConditionalGeneration is not None:
        return Qwen2VLForConditionalGeneration
    return Qwen2_5_VLForConditionalGeneration


def load_model_and_processor(base_model_path: str, device_ids: str, dtype: str = "auto", load_4bit: bool = False, load_8bit: bool = False, offline: bool = False):
    # Control offline/online behavior
    if offline:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
    else:
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        os.environ.pop("HF_HUB_OFFLINE", None)

    # Limit visible devices (enables multi-GPU when passing a comma list like "0,1")
    if device_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = device_ids
    # Let HF infer placement over the visible devices
    device_map = "auto"

    # Resolve torch dtype
    if dtype == "bfloat16":
        model_dtype = torch.bfloat16
    elif dtype == "float16":
        model_dtype = torch.float16
    else:
        model_dtype = "auto"

    # Build kwargs for quantization / dtype / device mapping
    pretrained_kwargs = {
        "torch_dtype": model_dtype,
        "device_map": device_map,
    }
    quantization_config = None
    if (load_4bit or load_8bit) and BitsAndBytesConfig is not None:
        if load_4bit and load_8bit:
            load_8bit = False
        if load_4bit:
            compute_dtype = torch.bfloat16 if dtype in ("auto", "bfloat16") else torch.float16
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
        elif load_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model_cls = _select_model_class(base_model_path)
    try:
        model = model_cls.from_pretrained(
            base_model_path,
            quantization_config=quantization_config,
            **pretrained_kwargs,
        )
    except Exception as e:
        # Try the other family if available
        alt_cls = Qwen2_5_VLForConditionalGeneration if model_cls is Qwen2VLForConditionalGeneration else Qwen2VLForConditionalGeneration
        if alt_cls is None:
            raise
        model = alt_cls.from_pretrained(
            base_model_path,
            quantization_config=quantization_config,
            **pretrained_kwargs,
        )
    processor = _load_processor_with_fallback(base_model_path)
    
    # Fix pad token warning
    if hasattr(processor, 'tokenizer') and processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    model.eval()
    return model, processor


def run_all_tasks(args):
    global FEWSHOT_EXAMPLES, FEWSHOT_VIDEOS
    
    root = Path(args.base_data_path)
    video_dir = root / "videos"
    
    # Load gold labels for few-shot selection
    gold_labels_path = root / "gold_labels.csv"
    if not gold_labels_path.exists():
        raise FileNotFoundError(f"Missing gold labels CSV: {gold_labels_path}")
    gold_labels = load_gold_labels(gold_labels_path)
    
    # Select random few-shot examples
    FEWSHOT_EXAMPLES = select_random_fewshot_examples(gold_labels, seed=args.seed)
    FEWSHOT_VIDEOS = {ex["video"] for ex in FEWSHOT_EXAMPLES}
    
    model, processor = load_model_and_processor(
        args.base_model_path,
        args.device_ids,
        dtype=args.torch_dtype,
        load_4bit=args.load_4bit,
        load_8bit=args.load_8bit,
        offline=args.offline,
    )

    # Load master list (single source of truth)
    ref_json = root / "video_mappings.json"
    if not ref_json.exists():
        raise FileNotFoundError(f"Missing master JSON: {ref_json}")
    items = load_json(ref_json)
    
    # Filter out few-shot examples from test set
    test_items = [item for item in items if item["video"] not in FEWSHOT_VIDEOS]
    print(f"Using {len(test_items)} test videos (excluding {len(FEWSHOT_VIDEOS)} few-shot examples)")
    
    # Build unique list of glosses for transparency_2 options
    all_glosses_unique = sorted({it.get("gloss", "") for it in items if it.get("gloss", "")})

    results: List[tuple] = []
    with torch.no_grad():
        for it in tqdm(test_items, desc="Processing all tasks"):
            vid = it["video"]
            gloss = it.get("gloss", "")
            video_path = str(video_dir / vid)
            
            # Handle both .mp4 and .mpeg extensions
            if not Path(video_path).exists():
                # Try switching extension
                if video_path.endswith('.mp4'):
                    alt_path = video_path.replace('.mp4', '.mpeg')
                elif video_path.endswith('.mpeg'):
                    alt_path = video_path.replace('.mpeg', '.mp4')
                else:
                    alt_path = None
                
                if alt_path and Path(alt_path).exists():
                    video_path = alt_path
                else:
                    print(f"Warning: Video file not found: {vid}")
                    continue

            dutch = ""
            category = ""
            # Read from master JSON item
            if isinstance(it, dict):
                dutch = it.get("dutch", "")
                category = it.get("category", "")

            for task in TASKS:
                # Use the new proper few-shot structure for all tasks
                messages = build_messages_for_video_with_fewshot(video_path, task, gloss, FEWSHOT_EXAMPLES, video_dir)
                
                # Handle transparency_2 special case by modifying the final question
                if task == "transparency_2":
                    # 10-option multiple choice: true gloss + 9 random distractors
                    distractors = [g for g in all_glosses_unique if g and g != gloss]
                    num_distractors = min(9, len(distractors))
                    sampled = random.sample(distractors, num_distractors) if num_distractors > 0 else []
                    options = [gloss] + sampled
                    random.shuffle(options)
                    options_str = ", ".join(options)
                    
                    # Replace the final question with the multiple choice version
                    final_question = (
                        "What does this sign resemble? Look at the form and movement of the sign. "
                        f"Choose the most likely option from these possibilities: {options_str}. "
                        "Answer with only the exact word from the list that best matches what the sign looks like."
                    )
                    
                    # Update the last message (test question)
                    messages[-1]["content"][1]["text"] = final_question
                
                # Process the messages using Qwen's vision processing
                image_inputs, video_inputs = process_vision_info(messages)
                try:
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                except Exception:
                    # Fallback to simple text if chat template fails
                    text = messages[-1]["content"][1]["text"]
                
                inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
                inputs = inputs.to(model.device)

                # Memory-efficient generation parameters for large models (72B+)
                is_large_model = "72B" in str(args.base_model_path) or "70B" in str(args.base_model_path)
                gen_ids = model.generate(
                    **inputs,
                    do_sample=False,
                    temperature=0.0,
                    max_new_tokens=128
                )
                
                trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, gen_ids)]
                output = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                if isinstance(output, list):
                    output = output[0]
                pred = (output or "").strip()
                print(f"{vid} | {task}: {pred}")
                results.append((vid, gloss, dutch, category, task, pred))

    model_name = Path(args.base_model_path).name
    out_csv = Path(args.results_path) / f"Iconicity_all_tasks_preds_fewshot_{model_name}.csv"
    save_unified_csv(results, out_csv)


def run_single_task(args):
    global FEWSHOT_EXAMPLES, FEWSHOT_VIDEOS
    
    root = Path(args.base_data_path)
    video_dir = root / "videos"
    master_path = root / "video_mappings.json"
    
    # Load gold labels for few-shot selection
    gold_labels_path = root / "gold_labels.csv"
    if not gold_labels_path.exists():
        raise FileNotFoundError(f"Missing gold labels CSV: {gold_labels_path}")
    gold_labels = load_gold_labels(gold_labels_path)
    
    # Select random few-shot examples
    FEWSHOT_EXAMPLES = select_random_fewshot_examples(gold_labels, seed=args.seed)
    FEWSHOT_VIDEOS = {ex["video"] for ex in FEWSHOT_EXAMPLES}

    model, processor = load_model_and_processor(
        args.base_model_path,
        args.device_ids,
        dtype=args.torch_dtype,
        load_4bit=args.load_4bit,
        load_8bit=args.load_8bit,
        offline=args.offline,
    )
    if not master_path.exists():
        raise FileNotFoundError(f"Missing master JSON: {master_path}")
    items = load_json(master_path)
    
    # Filter out few-shot examples from test set
    test_items = [item for item in items if item["video"] not in FEWSHOT_VIDEOS]
    print(f"Using {len(test_items)} test videos (excluding {len(FEWSHOT_VIDEOS)} few-shot examples)")

    rows: List[tuple] = []
    with torch.no_grad():
        for it in tqdm(test_items, desc=f"Predicting {args.task}"):
            vid = it["video"]
            gloss = it.get("gloss", "")
            video_path = str(video_dir / vid)
            
            # Handle both .mp4 and .mpeg extensions
            if not Path(video_path).exists():
                # Try switching extension
                if video_path.endswith('.mp4'):
                    alt_path = video_path.replace('.mp4', '.mpeg')
                elif video_path.endswith('.mpeg'):
                    alt_path = video_path.replace('.mpeg', '.mp4')
                else:
                    alt_path = None
                
                if alt_path and Path(alt_path).exists():
                    video_path = alt_path
                else:
                    print(f"Warning: Video file not found: {vid}")
                    continue

            # Use the new proper few-shot structure
            messages = build_messages_for_video_with_fewshot(video_path, args.task, gloss, FEWSHOT_EXAMPLES, video_dir)
            
            # Handle transparency_2 special case
            if args.task == "transparency_2":
                # Build the unique gloss list across items
                all_glosses_unique = sorted({jt.get("gloss", "") for jt in items if jt.get("gloss", "")})
                distractors = [g for g in all_glosses_unique if g and g != gloss]
                num_distractors = min(9, len(distractors))
                sampled = random.sample(distractors, num_distractors) if num_distractors > 0 else []
                options = [gloss] + sampled
                random.shuffle(options)
                options_str = ", ".join(options)
                
                # Replace the final question with the multiple choice version
                final_question = (
                    "What does this sign resemble? Look at the form and movement of the sign. "
                    f"Choose the most likely option from these possibilities: {options_str}. "
                    "Answer with only the exact word from the list that best matches what the sign looks like."
                )
                
                # Update the last message (test question)
                messages[-1]["content"][1]["text"] = final_question
            
            # Process the messages using Qwen's vision processing
            image_inputs, video_inputs = process_vision_info(messages)
            try:
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                # Fallback to simple text if chat template fails
                text = messages[-1]["content"][1]["text"]
            
            inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
            inputs = inputs.to(model.device)

            # Memory-efficient generation parameters for large models (72B+)
            is_large_model = "72B" in str(args.base_model_path) or "70B" in str(args.base_model_path)
            gen_ids = model.generate(
                **inputs,
                do_sample=False,
                temperature=0.0,
                max_new_tokens=128
            )
            
            trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, gen_ids)]
            output = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            if isinstance(output, list):
                output = output[0]
            pred = (output or "").strip()
            if args.task == "transparency_2":
                print(f"{vid} | transparency_2 answer: {pred}")
            rows.append((vid, args.task, gloss, pred))

    model_name = Path(args.base_model_path).name
    out_csv = Path(args.results_path) / f"Iconicity_{args.task}_preds_fewshot_{model_name}.csv"
    save_task_csv(rows, out_csv)


def parse_args():
    p = argparse.ArgumentParser(description="Qwen2.5-VL Iconicity few-shot inference")
    p.add_argument("--base_model_path", required=True, help="Local path to Qwen2.5-VL model (folder)")
    p.add_argument("--base_data_path", required=True, help="Path to data folder with JSON + videos/")
    p.add_argument("--results_path", default="results_v2", help="Output directory for CSV results")
    p.add_argument("--master_json", default="video_mappings.json", help="Unified master JSON with gloss/dutch/category")
    p.add_argument("--task", choices=TASKS, help="Run a single task")
    p.add_argument("--all_tasks", action="store_true", help="Run all tasks")
    p.add_argument("--device_ids", default="0", help="CUDA device IDs, e.g., '0' or '0,1'")
    p.add_argument("--torch_dtype", default="auto", choices=["auto", "bfloat16", "float16"], help="Torch dtype / precision")
    p.add_argument("--load_4bit", action="store_true", help="Load model in 4-bit quantization (bitsandbytes)")
    p.add_argument("--load_8bit", action="store_true", help="Load model in 8-bit quantization (bitsandbytes)")
    p.add_argument("--offline", action="store_true", help="Run fully offline (prevent downloads from HF Hub)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for few-shot example selection (default: 42)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.results_path, exist_ok=True)
    
    print(f"Using random seed: {args.seed}")
    print("Random 4-shot selection: 2 arbitrary, 1 high iconic, 1 low iconic\n")
    
    if args.all_tasks:
        run_all_tasks(args)
    elif args.task:
        run_single_task(args)
    else:
        raise SystemExit("Must specify either --task or --all_tasks")
