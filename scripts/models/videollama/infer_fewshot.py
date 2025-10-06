#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video-LLaMA 2 few-shot inference for Iconicity tasks.
Uses random 4-shot examples: 2 arbitrary, 1 high iconic, 1 low iconic.
"""

import os, json, csv, argparse, logging
from pathlib import Path
from tqdm import tqdm
import random
from typing import List, Dict

from videollama2 import model_init, mm_infer


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
    """Randomly select 8-shot examples: 4 arbitrary, 2 high iconic, 2 low iconic."""
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
    
    # Select 4 arbitrary examples
    if len(categories["arbitrary"]) >= 4:
        arbitrary_selected = random.sample(categories["arbitrary"], 4)
        selected_examples.extend(arbitrary_selected)
        for ex in arbitrary_selected:
            print(f"Selected arbitrary example: {ex['video']} - {ex['gloss']}")
    else:
        print(f"Warning: Not enough arbitrary examples (found {len(categories['arbitrary'])}, need 4)")
    
    # Select 2 high iconic examples
    if len(categories["high iconic"]) >= 2:
        high_iconic_selected = random.sample(categories["high iconic"], 2)
        selected_examples.extend(high_iconic_selected)
        for ex in high_iconic_selected:
            print(f"Selected high iconic example: {ex['video']} - {ex['gloss']}")
    else:
        print(f"Warning: Not enough high iconic examples (found {len(categories['high iconic'])}, need 2)")
    
    # Select 2 low iconic examples
    if len(categories["low iconic"]) >= 2:
        low_iconic_selected = random.sample(categories["low iconic"], 2)
        selected_examples.extend(low_iconic_selected)
        for ex in low_iconic_selected:
            print(f"Selected low iconic example: {ex['video']} - {ex['gloss']}")
    else:
        print(f"Warning: Not enough low iconic examples (found {len(categories['low iconic'])}, need 2)")
    
    return selected_examples


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
            "UMBRELLA", "WATER", "WHEELCHAIR", "WINDSCREEN", "ZIMMER"
        ]
        gloss_options = ", ".join(all_glosses)
        return f"What does this sign resemble? Look at the form and movement of the sign. Choose the most likely option from these possibilities: {gloss_options}. Answer with only the exact word from the list that best matches what the sign looks like."
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
    elif task == "reference":
        return example["reference"]
    elif task == "transparency":
        return example["transparency"]
    else:
        return "unknown"


def build_fewshot_prompt(task: str, test_gloss: str, fewshot_examples: List[Dict], video_dir: Path, processor) -> str:
    """Build few-shot prompt with examples for Video-LLaMA."""
    prompt_parts = []
    
    # Add few-shot examples
    for i, example in enumerate(fewshot_examples, 1):
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
        
        # Build question and answer for this example
        question = build_task_question(task, example["gloss"])
        answer = get_task_answer(task, example)
        
        prompt_parts.append(f"Example {i}: {question}")
        prompt_parts.append(f"Answer: {answer}")
        prompt_parts.append("")  # Empty line for separation
    
    # Add the test question
    test_question = build_task_question(task, test_gloss)
    prompt_parts.append("Now analyze this new sign:")
    prompt_parts.append(test_question)
    
    return "\n".join(prompt_parts)


def save_unified_csv(all_results, path: Path):
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


def save_task_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["video", "task", "gloss", "dutch", "category", "prediction"])
        wr.writerows(rows)


# ------------------------------- Inference ---------------------------------- #
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

    # Limit visible devices
    if args.device_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device_ids

    # Load model
    model_kwargs = {"device_map": "auto", "torch_dtype": args.torch_dtype}
    if args.load_4bit:
        model_kwargs["load_4bit"] = True
    if args.load_8bit:
        model_kwargs["load_8bit"] = True
    model, processor, tokenizer = model_init(args.base_model_path, **model_kwargs)

    # Master JSON
    ref_json = root / args.master_json
    if not ref_json.exists():
        raise FileNotFoundError(f"Missing master JSON: {ref_json}")
    items = load_json(ref_json)
    
    # Filter out few-shot examples from test set
    test_items = [item for item in items if item["video"] not in FEWSHOT_VIDEOS]
    print(f"Using {len(test_items)} test videos (excluding {len(FEWSHOT_VIDEOS)} few-shot examples)")
    
    # Build unique list of glosses for transparency_2 options
    all_glosses_unique = sorted({it.get("gloss", "") for it in items if it.get("gloss", "")})

    results = []
    for it in tqdm(test_items, desc="Processing all tasks"):
        vid = it["video"]
        gloss = it.get("gloss", "")
        dutch = it.get("dutch", "")
        category = it.get("category", "")
        
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
        
        frames = processor["video"](video_path)

        for task in TASKS:
            if task == "transparency_2":
                # 10 options: true gloss + 9 random distractors
                distractors = [g for g in all_glosses_unique if g and g != gloss]
                num_distractors = min(9, len(distractors))
                sampled = random.sample(distractors, num_distractors) if num_distractors > 0 else []
                options = [gloss] + sampled
                random.shuffle(options)
                gloss_options = ", ".join(options)
                
                # Build few-shot prompt with transparency_2 examples
                prompt_parts = []
                
                # Add few-shot examples
                for i, example in enumerate(FEWSHOT_EXAMPLES, 1):
                    question = f"What does this sign resemble? Look at the form and movement of the sign. Choose the most likely option from these possibilities: [example options]. Answer with only the exact word from the list that best matches what the sign looks like."
                    answer = example["transparency"]  # Use gloss as answer
                    
                    prompt_parts.append(f"Example {i}: {question}")
                    prompt_parts.append(f"Answer: {answer}")
                    prompt_parts.append("")  # Empty line for separation
                
                # Add the test question
                prompt_parts.append("Now analyze this new sign:")
                prompt_parts.append(
                    "What does this sign resemble? Look at the form and movement of the sign. "
                    f"Choose the most likely option from these possibilities: {gloss_options}. "
                    "Answer with only the exact word from the list that best matches what the sign looks like."
                )
                
                prompt = "\n".join(prompt_parts)
            else:
                # Build few-shot prompt for other tasks
                prompt = build_fewshot_prompt(task, gloss, FEWSHOT_EXAMPLES, video_dir, processor)
            
            pred = mm_infer(
                frames, 
                instruct=prompt, 
                model=model, 
                tokenizer=tokenizer, 
                modal="video",
                do_sample=False,
                temperature=0.0,
                max_new_tokens=128
            )
            print(f"{vid} | {task}: {pred}", flush=True)
            results.append((vid, gloss, dutch, category, task, pred))
    
    out_csv = Path(args.results_path) / f"Iconicity_all_tasks_preds_fewshot_{Path(args.base_model_path).name}.csv"
    save_unified_csv(results, out_csv)
    print(f"Saved unified results → {out_csv}")


def run_single_task(args):
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

    if args.device_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device_ids

    model_kwargs = {"device_map": "auto", "torch_dtype": args.torch_dtype}
    if args.load_4bit:
        model_kwargs["load_4bit"] = True
    if args.load_8bit:
        model_kwargs["load_8bit"] = True
    model, processor, tokenizer = model_init(args.base_model_path, **model_kwargs)

    ref_json = root / args.master_json
    if not ref_json.exists():
        raise FileNotFoundError(f"Missing master JSON: {ref_json}")
    items = load_json(ref_json)
    
    # Filter out few-shot examples from test set
    test_items = [item for item in items if item["video"] not in FEWSHOT_VIDEOS]
    print(f"Using {len(test_items)} test videos (excluding {len(FEWSHOT_VIDEOS)} few-shot examples)")

    rows = []
    for it in tqdm(test_items, desc=f"Predicting {args.task}"):
        vid = it["video"]
        gloss = it.get("gloss", "")
        dutch = it.get("dutch", "")
        category = it.get("category", "")
        
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
        
        frames = processor["video"](video_path)
        
        if args.task == "transparency_2":
            # Build the unique gloss list across items
            all_glosses_unique = sorted({jt.get("gloss", "") for jt in items if jt.get("gloss", "")})
            distractors = [g for g in all_glosses_unique if g and g != gloss]
            num_distractors = min(9, len(distractors))
            sampled = random.sample(distractors, num_distractors) if num_distractors > 0 else []
            options = [gloss] + sampled
            random.shuffle(options)
            gloss_options = ", ".join(options)
            
            # Build few-shot prompt with transparency_2 examples
            prompt_parts = []
            
            # Add few-shot examples
            for i, example in enumerate(FEWSHOT_EXAMPLES, 1):
                question = f"What does this sign resemble? Look at the form and movement of the sign. Choose the most likely option from these possibilities: [example options]. Answer with only the exact word from the list that best matches what the sign looks like."
                answer = example["transparency"]  # Use gloss as answer
                
                prompt_parts.append(f"Example {i}: {question}")
                prompt_parts.append(f"Answer: {answer}")
                prompt_parts.append("")  # Empty line for separation
            
            # Add the test question
            prompt_parts.append("Now analyze this new sign:")
            prompt_parts.append(
                "What does this sign resemble? Look at the form and movement of the sign. "
                f"Choose the most likely option from these possibilities: {gloss_options}. "
                "Answer with only the exact word from the list that best matches what the sign looks like."
            )
            
            prompt = "\n".join(prompt_parts)
        else:
            # Build few-shot prompt for other tasks
            prompt = build_fewshot_prompt(args.task, gloss, FEWSHOT_EXAMPLES, video_dir, processor)
        
        pred = mm_infer(
            frames, 
            instruct=prompt, 
            model=model, 
            tokenizer=tokenizer, 
            modal="video",
            do_sample=False,
            temperature=0.0,
            max_new_tokens=128
        )
        print(f"{vid} | {args.task}: {pred}", flush=True)
        rows.append((vid, args.task, gloss, dutch, category, pred))

    out_csv = Path(args.results_path) / f"Iconicity_{args.task}_preds_fewshot_{Path(args.base_model_path).name}.csv"
    save_task_csv(rows, out_csv)


def parse_args():
    p = argparse.ArgumentParser(description="Video-LLaMA 2 Iconicity few-shot inference")
    p.add_argument("--base_model_path", required=True, help="Local path to Video-LLaMA 2 model (folder)")
    p.add_argument("--base_data_path", required=True, help="Path to data folder with JSON + videos/")
    p.add_argument("--results_path", default="results_v2", help="Output directory for CSV results")
    p.add_argument("--master_json", default="video_mappings.json", help="Unified master JSON with gloss/dutch/category")
    p.add_argument("--task", choices=TASKS, help="Run a single task")
    p.add_argument("--torch_dtype", default="float16", choices=["auto", "float16", "bfloat16"], help="Torch dtype / precision")
    p.add_argument("--all_tasks", action="store_true", help="Run all tasks")
    p.add_argument("--device_ids", default="0", help="CUDA device IDs, e.g., '0' or '0,1'")
    p.add_argument("--load_4bit", action="store_true", help="Load model in 4-bit quantization")
    p.add_argument("--load_8bit", action="store_true", help="Load model in 8-bit quantization")
    p.add_argument("--seed", type=int, default=42, help="Random seed for few-shot example selection (default: 42)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.results_path, exist_ok=True)
    
    print(f"Using random seed: {args.seed}")
    print("Random 8-shot selection: 4 arbitrary, 2 high iconic, 2 low iconic\n")
    
    if args.all_tasks:
        run_all_tasks(args)
    elif args.task:
        run_single_task(args)
    else:
        raise SystemExit("Must specify either --task or --all_tasks")
