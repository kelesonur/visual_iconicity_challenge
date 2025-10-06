#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT zero-shot inference (8-frame sampling) via OpenAI Responses API
for benchmarking iconicity/phonology on sign-language clips.

Usage:
  python infer.py --base_data_path data --all_tasks --frame_count 8
  python infer.py --base_data_path data --task handshape --frame_count 8
  python infer.py --base_data_path data --model_name gpt-4o --all_tasks --frame_count 8
"""

import os
import csv
import json
import time
import math
import base64
import argparse
import random
import hashlib
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from tqdm import tqdm
from openai import OpenAI

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

# ----------------------------- Prompt builders ----------------------------- #
def build_iconicity_prompt(task: str, meaning: str = "") -> str:
    if task == "iconicity_rating":
        return (
            f"This sign means: {meaning}. "
            "Some signs are iconic and some are arbitrary. "
            "Find visual resemblances between the meaning and the form of the sign. "
            f"How much does the sign look like '{meaning}'? "
            "Answer with only one number: 1,2,3,4,5,6,7 (1=not at all, 7=exactly)."
        )
    if task == "handshape":
        return (
            "Major sign handshape? Answer with only one: H1,H2,H3,H4,H5,H6,H7 "
            "(H1=all fingers closed to a fist, H2=all fingers extended, H3=all fingers curved or clawed, "
            "H4=one (selected) finger extended, H5=one (selected) finger curved or clawed, "
            "H6=two or more (selected) fingers extended, H7=two or more (selected) fingers curved or clawed)."
        )
    if task == "location":
        return (
            "Major sign location? Answer with only one: L1,L2,L3,L4,L5 "
            "(L1=hands touching head/face, L2=hands touching torso, L3=hands touching arm, "
            "L4=hands touching weak/passive hand, L5=hands in front of the body or face)."
        )
    if task == "path_shape":
        return (
            "Movement path shape? Answer with only one: Hold, Straight, Arched, Circular. "
            "(Hold=no path or direction, Straight=move in a straight line, Arched=move in an arched line, "
            "Circular=move in a circular path)."
        )
    if task == "path_repetition":
        return (
            "Movement repetition? Answer with only one: Single, Repeated. "
            "(Single=one movement, Repeated=multiple or repeated movements)."
        )
    if task == "handedness":
        return (
            "Handedness? Answer with only one: One-handed, Two-handed symmetrical, Two-handed asymmetrical. "
            "(One-handed=only one hand is used in the sign, Two-handed symmetrical=two hands move together "
            "with the same handshape, Two-handed asymmetrical=two hands visible but one stationary and handshapes differ)."
        )
    if task == "transparency":
        all_glosses = [
            "AMBULANCE","BABY","BALL","BEAR","BED","BICYCLE","BIRD","BLANKET","BOTTLE","BOX",
            "BRIDGE","BUILDING","BUS","BUTTERFLY","CALCULATOR","CAMEL","CAR","CAT","CELL","CHICKEN",
            "CURTAINS","DEER","DOCTOR","DOG","DOOR","DRILL","EAGLE","ELECTRICITY","ELEPHANT","FIRE",
            "FRUIT","HELICOPTER","HORSE","INTERPRETER","KEY","KIWI","LAPTOP","LIGHTBULB","LOBSTER","MONKEY",
            "MUMMY","PENGUIN","PERSON","PIANO","PISTOL","PLANE","PRAM","PUPPET","RABBIT","RATTLE",
            "RESTAURANT","ROOM","SHEEP","SNAKE","SOFA","SPIDER","SPOON","SUGAR","SUITCASE","TABLE",
            "TELEPHONE","TO-ARGUE","TO-BREAK","TO-COOK","TO-CRASH","TO-CRY","TO-CUT","TO-DIE","TO-ERASE","TO-FLY",
            "TO-GO-OUT","TO-GOSSIP","TOILET","TO-INJECT","TO-JUGGLE","TO-KNOCK","TO-LAUGH","TO-ORDER","TO-PLAY-CARDS","TO-PUMP",
            "TO-PUT-CLOTHES-ON","TO-SHOUT","TO-SLAP","TO-SMS","TO-STAPLE","TO-STEAL","TO-SWIM","TOWEL","TO-WRING","TREE",
            "UMBRELLA","WATER","WHEELCHAIR","WINDSCREEN WIPER","ZIMMER"
        ]
        gloss_options = ", ".join(all_glosses)
        return (
            "What does this sign resemble? Look at the form and movement of the sign. "
            f"Choose the most likely option from these possibilities: {gloss_options}. "
            "Answer with only the exact word from the list that best matches what the sign looks like. "
            "If the sign does not resemble any of the above, answer 'UNKNOWN'."
        )
    return f"Analyze the sign's {task.replace('_', ' ')} and answer with the single required label."

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
        by_vid: Dict[str, Dict] = {}
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

# ----------------------------- Video → frames -------------------------------- #
def ffprobe_duration(video_path: str) -> Optional[float]:
    try:
        out = subprocess.check_output([
            "ffprobe","-v","error","-show_entries","format=duration",
            "-of","default=noprint_wrappers=1:nokey=1", video_path
        ], stderr=subprocess.STDOUT).decode().strip()
        return float(out)
    except Exception:
        return None

def extract_n_frames(video_path: str, out_root: Path, n: int = 8, width: int = 512) -> List[str]:
    """
    Evenly sample ~n frames across duration using fps ≈ n/duration. Returns PNG paths in order.
    Cached under results/_frames_cache/<stem>_<hash>/frame_XXX.png
    """
    out_root.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5((video_path + f"|{n}|{width}").encode("utf-8")).hexdigest()[:10]
    cache_dir = out_root / f"{Path(video_path).stem}_{key}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(cache_dir.glob("frame_*.png"))
    if existing:
        return [str(p) for p in existing]

    dur = ffprobe_duration(video_path)
    fps = max(0.5, n / dur) if (dur and dur > 0) else 2.0
    try:
        subprocess.run([
            "ffmpeg","-y","-i",video_path,
            "-vf", f"fps={fps},scale={width}:-1:flags=lanczos",
            str(cache_dir / "frame_%03d.png")
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as e:
        print(f"Frame extraction failed for {video_path}: {e}")
        return []

    frames = sorted(cache_dir.glob("frame_*.png"))
    if len(frames) > n and n > 0:
        # downsample evenly to exactly n frames
        idxs = [math.floor(i * (len(frames) / n)) for i in range(n)]
        frames = [frames[i] for i in idxs]
        # rewrite to fixed names to preserve order
        for i, p in enumerate(frames):
            newp = cache_dir / f"frame_{i:03d}.png"
            if p.name != newp.name:
                p.rename(newp)
        frames = sorted(cache_dir.glob("frame_*.png"))
    return [str(p) for p in frames]

# ----------------------------- OpenAI helpers -------------------------------- #
class GPTClient:
    """
    Wraps OpenAI SDK. Encodes frame images as base64 and caches for reuse.
    Uses the Responses API for GPT-5 models.
    """
    def __init__(self, model_name: str = "gpt-5"):
        self.client = OpenAI()
        self.model_name = model_name
        self._image_cache: Dict[str, str] = {}  # path -> base64_url

    def encode_image(self, image_path: str) -> Optional[str]:
        """Encode image to base64 data URL, with caching."""
        if image_path in self._image_cache:
            return self._image_cache[image_path]
        try:
            with open(image_path, "rb") as img_file:
                encoded = base64.b64encode(img_file.read()).decode('utf-8')
                data_url = f"data:image/png;base64,{encoded}"
                self._image_cache[image_path] = data_url
                return data_url
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None

    def ask_with_images(self, image_paths: List[str], instruction: str, max_retries: int = 3) -> str:
        """
        Send multiple ordered frames + instruction using Responses API. Order matters (temporal).
        """
        for attempt in range(max_retries):
            try:
                # Build content array with images and text instruction
                content = []
                for img_path in image_paths:
                    img_url = self.encode_image(img_path)
                    if not img_url:
                        print(f"Failed to encode image: {img_path}")
                        continue
                    content.append({
                        "type": "input_image",
                        "image_url": img_url
                    })
                
                content.append({
                    "type": "input_text",
                    "text": (
                        "You are a precise assistant that analyzes an ordered sequence of frames "
                        "from a short sign language clip. "
                        "Output only the required label with no extra words. "
                        + instruction
                    )
                })
                
                # Standardized generation parameters
                resp = self.client.responses.create(
                    model=self.model_name,
                    input=[{"role": "user", "content": content}],
                    max_output_tokens=128,
                    metadata={"purpose": "academic_research"}
                )
                
                text = (resp.output_text or "").strip()
                if text:
                    return text

                print(f"Empty response from GPT (attempt {attempt + 1}/{max_retries})", flush=True)
                
            except Exception as e:
                print(f"Error from GPT (attempt {attempt+1}/{max_retries}): {e}", flush=True)
            time.sleep(2)
        return ""

# ----------------------------- Core runners ---------------------------------- #
def resolve_video_path(video_dir: Path, vid: str) -> Optional[str]:
    p = video_dir / vid
    if p.exists():
        return str(p)
    s = str(p)
    alt = None
    if s.endswith(".mp4"):
        alt = s[:-4] + ".mpeg"
    elif s.endswith(".mpeg"):
        alt = s[:-5] + ".mp4"
    if alt and Path(alt).exists():
        return alt
    print(f"Warning: Video file not found: {vid}", flush=True)
    return None

def run_all_tasks(args):
    root = Path(args.base_data_path)
    video_dir = root / "videos"
    master_json = Path(args.base_data_path) / args.master_json
    if not master_json.exists():
        raise FileNotFoundError(f"Missing master JSON: {master_json}")
    items = load_json(master_json)

    # Unique glosses for transparency_2
    all_glosses_unique = sorted({it.get("gloss", "") for it in items if it.get("gloss", "")})

    if args.limit:
        items = items[:args.limit]

    gpt = GPTClient(model_name=args.model_name)
    results: List[tuple] = []
    frames_cache_root = Path(args.results_path) / "_frames_cache"

    for it in tqdm(items, desc="Processing all tasks"):
        vid = it["video"]
        gloss = it.get("gloss", "")
        dutch = it.get("dutch", "")
        category = it.get("category", "")

        video_path = resolve_video_path(video_dir, vid)
        if not video_path:
            continue

        # extract frames
        frames = extract_n_frames(video_path, frames_cache_root, n=args.frame_count, width=512)
        if not frames:
            print(f"Skipping {vid} due to frame extraction failure", flush=True)
            continue

        for task in TASKS:
            if task == "transparency_2":
                # true gloss + 9 random distractors
                distractors = [g for g in all_glosses_unique if g and g != gloss]
                k = min(9, len(distractors))
                sampled = random.sample(distractors, k) if k > 0 else []
                options = [gloss] + sampled
                random.shuffle(options)
                instruction = (
                    "What does this sign resemble? Look at the form and movement across the frames. "
                    f"Choose the most likely option from these possibilities: {', '.join(options)}. "
                    "Answer with only the exact word from the list."
                )
            else:
                instruction = build_iconicity_prompt(task, gloss)

            # Zero-shot inference
            pred = gpt.ask_with_images(frames, instruction)
            print(f"{vid} | {task}: {pred}", flush=True)
            results.append((vid, gloss, dutch, category, task, pred))

    out_csv = Path(args.results_path) / f"Iconicity_all_tasks_preds_{args.model_name}.csv"
    save_unified_csv(results, out_csv)
    print(f"Wrote {out_csv}", flush=True)

def run_single_task(args):
    root = Path(args.base_data_path)
    video_dir = root / "videos"
    master_json = Path(args.base_data_path) / args.master_json
    if not master_json.exists():
        raise FileNotFoundError(f"Missing master JSON: {master_json}")
    items = load_json(master_json)
    
    if args.limit:
        items = items[:args.limit]

    # for transparency_2
    all_glosses_unique = sorted({jt.get("gloss", "") for jt in items if jt.get("gloss", "")})

    gpt = GPTClient(model_name=args.model_name)
    rows: List[tuple] = []
    frames_cache_root = Path(args.results_path) / "_frames_cache"

    for it in tqdm(items, desc=f"Predicting {args.task}"):
        vid = it["video"]
        gloss = it.get("gloss", "")

        if args.task == "transparency_2":
            distractors = [g for g in all_glosses_unique if g and g != gloss]
            k = min(9, len(distractors))
            sampled = random.sample(distractors, k) if k > 0 else []
            options = [gloss] + sampled
            random.shuffle(options)
            instruction = (
                "What does this sign resemble? Look at the form and movement across the frames. "
                f"Choose the most likely option from these possibilities: {', '.join(options)}. "
                "Answer with only the exact word from the list."
            )
        else:
            instruction = build_iconicity_prompt(args.task, gloss)

        video_path = resolve_video_path(video_dir, vid)
        if not video_path:
            continue

        frames = extract_n_frames(video_path, frames_cache_root, n=args.frame_count, width=512)
        if not frames:
            print(f"Skipping {vid} due to frame extraction failure", flush=True)
            continue

        # Zero-shot inference
        pred = gpt.ask_with_images(frames, instruction)
        print(f"{vid} | {args.task}: {pred}", flush=True)
        rows.append((vid, args.task, gloss, pred))

    out_csv = Path(args.results_path) / f"Iconicity_{args.task}_preds_{args.model_name}.csv"
    save_task_csv(rows, out_csv)
    print(f"Wrote {out_csv}", flush=True)

# ----------------------------- CLI ------------------------------------------ #
def parse_args():
    p = argparse.ArgumentParser(description="GPT zero-shot inference with 8-frame sampling (OpenAI Responses API)")
    p.add_argument("--model_name", default="gpt-4o", help="Model name (e.g., gpt-5, gpt-4o, gpt-4-turbo)")
    p.add_argument("--base_data_path", required=True, help="Path with JSON + videos/")
    p.add_argument("--results_path", default="results", help="Output directory for CSV results")
    p.add_argument("--master_json", default="video_mappings.json", help="Unified master JSON with gloss/dutch/category")
    p.add_argument("--task", choices=TASKS, help="Run a single task")
    p.add_argument("--all_tasks", action="store_true", help="Run all tasks")
    p.add_argument("--limit", type=int, help="Limit number of videos (useful for testing)")
    p.add_argument("--frame_count", type=int, default=8, help="Number of frames to sample per video")
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
