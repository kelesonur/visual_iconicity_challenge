#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video-LLaMA 2 zero-shot inference for Iconicity tasks.
Uses the same tasks and exact prompts as Qwen `infer_minimal.py`.
"""

import os, json, csv, argparse, logging
from pathlib import Path
from tqdm import tqdm
import random

from videollama2 import model_init, mm_infer


# ----------------------------- Tasks (minimal prompts) ----------------------------- #
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


def build_iconicity_prompt(task: str, meaning: str = "") -> str:
    # Iconicity (binary)
    if task == "iconicity_binary":
        return (
            f"Meaning: {meaning}. "
            "Some signs are iconic and some are arbitrary."
            "Find visual resemblances between the meaning and the form of the sign. "
            f"Does the sign look like or resemble '{meaning}'? Answer only one word: yes or no"
        )

    # Iconicity Rating (1-7)
    if task == "iconicity_rating":
        return (
            f"This sign means: {meaning}. "
            "Some signs are iconic and some are arbitrary."
            "Find visual resemblances between the meaning and the form of the sign. "
            f"How much does the sign look like '{meaning}'? "
            "Answer with only one number: 1,2,3,4,5,6,7 (1=not at all, 7=exactly)."
        )

    # Handshape
    if task == "handshape":
        return (
            "Major sign handshape? Answer with only one: H1,H2,H3,H4,H5,H6,H7"
            "(H1=all fingers closed to a fist, H2=all fingers extended, H3=all fingers curved or clawed, H4=one (selected) finger extended, H5=one (selected) finger curved or clawed, H6= two or more (selected) fingers extended, H7=two or more(selected) fingers curved or clawed)."
        )

    # Location
    if task == "location":
        return (
            "Major sign location? Answer with only one: L1, L2, L3, L4, L5"
            "(L1=hands touching head/face, L2=hands touching torso, L3=hands touching arm, L4=hands touching weak/passive hand, L5=hands in front of the body or face)"
        )

    # Path shape
    if task == "path_shape":
        return (
            "Movement path shape? Answer with only one: Hold, Straight, Arched, Circular."
            "(Hold=no path or direction, Straight=move in a straight line, Arched=move in an arched line, Circular=move in a circular path)"
        )

    # Path repetition
    if task == "path_repetition":
        return (
            "Movement repetition? Answer with only one: Single, Repeated."
            "(Single=one movement, Repeated=multiple or repeated movements)"
        )

    # Handedness
    if task == "handedness":
        return (
            "Handedness? Answer with only one: One-handed, Two-handed symmetrical, Two-handed asymmetrical."
            "(One-handed=only one hand is used in the sign, Two-handed symmetrical=two hands are used but the hands move together and have the same handshape, Two-handed asymmetrical=two hands are visible, but one hand does not move and the hands have different handshapes)"
        )

    # Reference
    if task == "reference":
        return (    
            "Reference? Answer with only one: Object, Action, Other."
            "(Object=the sign resembles an object, Action=the sign resembles an action, Other=the sign does not resemble any of the above)"
        )

    # Transparency (classification from all possible glosses)
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
            "UMBRELLA", "WATER", "WHEELCHAIR", "WINDSCREEN", "ZIMMER"
        ]
        
        gloss_options = ", ".join(all_glosses)
        return (
            "What does this sign resemble? Look at the form and movement of the sign. "
            f"Choose the most likely option from these possibilities: {gloss_options}. "
            "Answer with only the exact word from the list that best matches what the sign looks like."
        )

    # Fallback
    return f"Analyze the sign's {task.replace('_', ' ')}."


# ----------------------------- I/O utilities -------------------------------- #
def load_json(path: Path):
    with path.open() as fh:
        return json.load(fh)


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
    root = Path(args.base_data_path)
    video_dir = root / "videos"

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

    results = []
    for it in tqdm(items, desc="Processing all tasks"):
        vid = it["video"]
        gloss = it.get("gloss", "")
        dutch = it.get("dutch", "")
        category = it.get("category", "")
        frames = processor["video"](str(video_dir / vid))

        for task in TASKS:
            if task == "transparency_2":
                # 10 options: true gloss + 9 random distractors
                all_glosses = [x.get("gloss", "") for x in items if x.get("gloss", "")]
                distractors = [g for g in all_glosses if g and g != gloss]
                num_distractors = min(9, len(distractors))
                sampled = random.sample(distractors, num_distractors) if num_distractors > 0 else []
                options = [gloss] + sampled
                random.shuffle(options)
                gloss_options = ", ".join(options)
                prompt = (
                    "What does this sign resemble? Look at the form and movement of the sign. "
                    f"Choose the most likely option from these possibilities: {gloss_options}. "
                    "Answer with only the exact word from the list that best matches what the sign looks like."
                )
            else:
                prompt = build_iconicity_prompt(task, gloss)
            if task == "transparency_2":
                print(f"{vid} | transparency_2 prompt: {prompt}")
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
            if task == "transparency_2":
                print(f"{vid} | transparency_2 answer: {pred}")
            print(f"{vid} | {task}: {pred}", flush=True)
            results.append((vid, gloss, dutch, category, task, pred))
    out_csv = Path(args.results_path) / f"Iconicity_all_tasks_preds_{Path(args.base_model_path).name}.csv"
    save_unified_csv(results, out_csv)
    print(f"Saved unified results → {out_csv}")


def run_single_task(args):
    root = Path(args.base_data_path)
    video_dir = root / "videos"

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

    rows = []
    for it in tqdm(items, desc=f"Predicting {args.task}"):
        vid = it["video"]
        gloss = it.get("gloss", "")
        dutch = it.get("dutch", "")
        category = it.get("category", "")
        frames = processor["video"](str(video_dir / vid))
        if args.task == "transparency_2":
            all_glosses = [x.get("gloss", "") for x in items if x.get("gloss", "")]
            distractors = [g for g in all_glosses if g and g != gloss]
            num_distractors = min(9, len(distractors))
            sampled = random.sample(distractors, num_distractors) if num_distractors > 0 else []
            options = [gloss] + sampled
            random.shuffle(options)
            gloss_options = ", ".join(options)
            prompt = (
                "What does this sign resemble? Look at the form and movement of the sign. "
                f"Choose the most likely option from these possibilities: {gloss_options}. "
                "Answer with only the exact word from the list that best matches what the sign looks like."
            )
        else:
            prompt = build_iconicity_prompt(args.task, gloss)
        if args.task == "transparency_2":
            print(f"{vid} | transparency_2 prompt: {prompt}")
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
        if args.task == "transparency_2":
            print(f"{vid} | transparency_2 answer: {pred}")
        print(f"{vid} | {args.task}: {pred}", flush=True)
        rows.append((vid, args.task, gloss, dutch, category, pred))

    out_csv = Path(args.results_path) / f"Iconicity_{args.task}_preds_{Path(args.base_model_path).name}.csv"
    save_task_csv(rows, out_csv)


def parse_args():
    p = argparse.ArgumentParser(description="Video-LLaMA 2 Iconicity inference (local)")
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
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.results_path, exist_ok=True)
    if args.all_tasks:
        run_all_tasks(args)
    elif args.task:
        run_single_task(args)
    else:
        raise SystemExit("Must specify either --task or --all_tasks")
