#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gemini Pro 2.5 zero-shot inference via Vertex AI for seven tasks for benchmarking iconicity."""

import os
import sys
import csv
import json
import argparse
from pathlib import Path
from typing import List, Tuple
import random
import time

from tqdm import tqdm
import vertexai
from vertexai.generative_models import GenerativeModel, Part, HarmCategory, HarmBlockThreshold


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


def build_iconicity_prompt(task: str, meaning: str = "") -> str:
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
            "UMBRELLA", "WATER", "WHEELCHAIR", "WINDSCREEN WIPER", "ZIMMER"
        ]
        
        gloss_options = ", ".join(all_glosses)
        return (
            "What does this sign resemble? Look at the form and movement of the sign. "
            f"Choose the most likely option from these possibilities: {gloss_options}. "
            "Answer with only the exact word from the list that best matches what the sign looks like."
            "If the sign does not resemble any of the above, answer 'UNKNOWN'"
        )

    # Fallback
    return f"Analyze the sign's {task.replace('_', ' ')}."


# ----------------------------- I/O utilities -------------------------------- #
def load_json(path: Path):
    with path.open() as fh:
        return json.load(fh)


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


# ------------------------------- Video Processing ---------------------------------- #
def load_video_part(video_path: str) -> Part:
    """Load video file as a Part object for Vertex AI."""
    try:
        with open(video_path, 'rb') as f:
            video_data = f.read()
        
        # Determine MIME type based on extension
        ext = Path(video_path).suffix.lower()
        if ext == '.mp4':
            mime_type = 'video/mp4'
        elif ext in ['.mpeg', '.mpg']:
            mime_type = 'video/mpeg'
        elif ext == '.mov':
            mime_type = 'video/quicktime'
        elif ext == '.avi':
            mime_type = 'video/x-msvideo'
        else:
            mime_type = 'video/mp4'  # default
        
        return Part.from_data(data=video_data, mime_type=mime_type)
    except Exception as e:
        print(f"Error loading video {video_path}: {e}")
        return None


# ------------------------------- Inference ---------------------------------- #
def query_gemini_with_video(model, video_part, instruction: str, max_retries: int = 3) -> str:
    """Query Gemini model via Vertex AI with video and instruction."""
    # Configure safety settings to be more permissive
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    }
    
    for attempt in range(max_retries):
        try:
            # Build prompt with video and instruction
            prompt = f"You are a helpful assistant that analyzes sign language videos. {instruction}"
            
            # Generate response
            response = model.generate_content(
                [video_part, prompt],
                generation_config={
                    "temperature": 0.0,
                    "max_output_tokens": 128,
                },
                safety_settings=safety_settings
            )
            
            if response and response.text:
                return response.text.strip()
            else:
                print(f"Empty response from Gemini (attempt {attempt + 1}/{max_retries})")
                time.sleep(2)
        except Exception as e:
            print(f"Error querying Gemini (attempt {attempt + 1}/{max_retries}): {e}")
            time.sleep(2)
    
    return ""


def initialize_vertex_ai(project_id: str, location: str, model_name: str = "gemini-2.0-flash-exp"):
    """Initialize Vertex AI and return the Gemini model."""
    vertexai.init(project=project_id, location=location)
    
    # Available models: gemini-2.0-flash-exp, gemini-1.5-pro, gemini-1.5-flash, gemini-pro, etc.
    model = GenerativeModel(model_name)
    return model


def run_all_tasks(args):
    root = Path(args.base_data_path)
    video_dir = root / "videos"
    
    # Initialize Vertex AI
    project_id = args.project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        raise ValueError("Project ID must be provided via --project_id or GOOGLE_CLOUD_PROJECT environment variable")
    
    model = initialize_vertex_ai(project_id, args.location, args.model_name)

    # Load master list (single source of truth)
    ref_json = root / "video_mappings.json"
    if not ref_json.exists():
        raise FileNotFoundError(f"Missing master JSON: {ref_json}")
    items = load_json(ref_json)
    # Build unique list of glosses for transparency_2 options
    all_glosses_unique = sorted({it.get("gloss", "") for it in items if it.get("gloss", "")})

    # Limit items if specified
    if hasattr(args, 'limit') and args.limit:
        items = items[:args.limit]
    
    results: List[tuple] = []
    for it in tqdm(items, desc="Processing all tasks"):
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

        # Load video once for all tasks
        video_part = load_video_part(video_path)
        if not video_part:
            print(f"Skipping {vid} due to video loading failure")
            continue

        for task in TASKS:
            if task == "transparency_2":
                # 10-option multiple choice: true gloss + 9 random distractors
                distractors = [g for g in all_glosses_unique if g and g != gloss]
                num_distractors = min(9, len(distractors))
                sampled = random.sample(distractors, num_distractors) if num_distractors > 0 else []
                options = [gloss] + sampled
                random.shuffle(options)
                options_str = ", ".join(options)
                instruction = (
                    "What does this sign resemble? Look at the form and movement of the sign. "
                    f"Choose the most likely option from these possibilities: {options_str}. "
                    "Answer with only the exact word from the list that best matches what the sign looks like."
                )
            else:
                instruction = build_iconicity_prompt(task, gloss)
            
            pred = query_gemini_with_video(model, video_part, instruction)
            print(f"{vid} | {task}: {pred}")
            results.append((vid, gloss, dutch, category, task, pred))

    out_csv = Path(args.results_path) / f"Iconicity_all_tasks_preds_{args.model_name}.csv"
    save_unified_csv(results, out_csv)


def run_single_task(args):
    root = Path(args.base_data_path)
    video_dir = root / "videos"
    master_path = root / "video_mappings.json"

    # Initialize Vertex AI
    project_id = args.project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        raise ValueError("Project ID must be provided via --project_id or GOOGLE_CLOUD_PROJECT environment variable")
    
    model = initialize_vertex_ai(project_id, args.location, args.model_name)

    if not master_path.exists():
        raise FileNotFoundError(f"Missing master JSON: {master_path}")
    items = load_json(master_path)

    # Limit items if specified
    if hasattr(args, 'limit') and args.limit:
        items = items[:args.limit]

    rows: List[tuple] = []
    for it in tqdm(items, desc=f"Predicting {args.task}"):
        vid = it["video"]
        gloss = it.get("gloss", "")
        if args.task == "transparency_2":
            # 10-option multiple choice: true gloss + 9 random distractors
            # Build the unique gloss list across items
            all_glosses_unique = sorted({jt.get("gloss", "") for jt in items if jt.get("gloss", "")})
            distractors = [g for g in all_glosses_unique if g and g != gloss]
            num_distractors = min(9, len(distractors))
            sampled = random.sample(distractors, num_distractors) if num_distractors > 0 else []
            options = [gloss] + sampled
            random.shuffle(options)
            options_str = ", ".join(options)
            instruction = (
                "What does this sign resemble? Look at the form and movement of the sign. "
                f"Choose the most likely option from these possibilities: {options_str}. "
                "Answer with only the exact word from the list that best matches what the sign looks like."
            )
        else:
            instruction = build_iconicity_prompt(args.task, gloss)
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

        # Load video
        video_part = load_video_part(video_path)
        if not video_part:
            print(f"Skipping {vid} due to video loading failure")
            continue

        pred = query_gemini_with_video(model, video_part, instruction)
        if args.task == "transparency_2":
            print(f"{vid} | transparency_2 answer: {pred}")
        rows.append((vid, args.task, gloss, pred))

    out_csv = Path(args.results_path) / f"Iconicity_{args.task}_preds_{args.model_name}.csv"
    save_task_csv(rows, out_csv)


def parse_args():
    p = argparse.ArgumentParser(description="Gemini Pro 2.5 Iconicity inference via Vertex AI")
    p.add_argument("--project_id", help="Google Cloud project ID (or set GOOGLE_CLOUD_PROJECT env var)")
    p.add_argument("--location", default="us-central1", help="Vertex AI location/region (e.g., us-central1, europe-west1)")
    p.add_argument("--model_name", default="gemini-2.0-flash-exp", help="Gemini model name (e.g., gemini-2.0-flash-exp, gemini-1.5-pro, gemini-pro)")
    p.add_argument("--base_data_path", required=True, help="Path to data folder with JSON + videos/")
    p.add_argument("--results_path", default="results_v2", help="Output directory for CSV results")
    p.add_argument("--master_json", default="video_mappings.json", help="Unified master JSON with gloss/dutch/category")
    p.add_argument("--task", choices=TASKS, help="Run a single task")
    p.add_argument("--all_tasks", action="store_true", help="Run all tasks")
    p.add_argument("--limit", type=int, help="Limit number of videos to process (useful for testing)")
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

