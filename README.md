# The Visual Iconicity Challenge

**Evaluating Vision-Language Models on Sign Language Form–Meaning Mapping**

A video-based benchmark that adapts psycholinguistic measures to evaluate whether vision-language models (VLMs) can recover form-meaning mappings from dynamic human motion in Sign Language of the Netherlands (NGT). The benchmark tests VLMs on three tasks: phonological form prediction, transparency (meaning inference from visual form), and graded iconicity ratings.

## Dataset

**96 NGT signs** with annotations:
- **64 iconic signs** (high human iconicity ratings)
- **32 arbitrary signs** (low human iconicity ratings)
- English glosses (meanings)
- Human iconicity ratings (1-7 scale, crowdsourced)
- Ground-truth phonological features (5 parameters)
- Human baselines from deaf signer and sign-naive participants

Based on Ortega et al. (2019) with extensions for this benchmark.

**Note**: Sign videos are not included in this repository. Please refer to the original dataset source.

## Quick Start

```bash
# Zero-shot inference example
python scripts/models/qwen/infer_zeroshot.py \
  --model_path Qwen/Qwen2.5-VL-7B-Instruct \
  --dataset_path data/gold_labels.csv \
  --output_path results/qwen_zeroshot.csv

# Few-shot inference example  
python scripts/models/qwen/infer_fewshot.py \
  --model_path Qwen/Qwen2.5-VL-7B-Instruct \
  --dataset_path data/gold_labels.csv \
  --output_path results/qwen_fewshot.csv
```

## Tasks

### 1. Phonological Form Prediction
Predict 5 phonological features from sign videos:
- **Handshape** (7 categories)
- **Location** (5 categories) 
- **Path Shape** (4 categories)
- **Path Repetition** (2 categories)
- **Handedness** (3 categories)

**Metric**: Accuracy per feature and mean accuracy across features

### 2. Transparency
Infer sign meaning from visual form alone:
- **Open-set** (96 options): identify correct gloss among all 96 signs
- **Multiple-choice** (10 options): select from 10 candidates

**Metric**: Accuracy

### 3. Graded Iconicity Rating
Rate form-meaning resemblance on 1-7 scale (1=not iconic, 7=exactly resembles meaning)

**Metrics**: Spearman's ρ correlation with human ratings, Cohen's d for iconic vs. arbitrary separation

## Key Findings

**Phonological Form**: VLMs recover some handshape and location detail but remain below human performance (best: Gemini 2.5 Pro 0.706 vs. human 0.794). Models mirror human acquisition patterns: location > handshape.

**Transparency**: Large gap from human baselines. Best model (Gemini 2.5 Pro) identifies only 17/96 signs vs. 57/96 (deaf signer) and 40/96 (sign-naive). Models succeed mainly on visually obvious signs.

**Iconicity**: Top models show moderate correlation with human ratings (GPT-5 ρ=0.607, Gemini 2.5 Pro ρ=0.577, Qwen2.5-VL-72B ρ=0.501). Models with stronger phonological prediction correlate better with human iconicity judgments.

**Few-shot effects**: Modest gains on phonology and iconicity for some models; no improvement on transparency.

**Type bias**: Humans prefer action-based iconic signs; most VLMs favor object-based signs, revealing reliance on static visual resemblance over dynamic action mappings.

## Repository Structure

```
visual_iconicity_challenge/
├── data/
│   ├── gold_labels.csv           # Annotations and human ratings
│   └── video_mappings.json       # Video file mappings (videos not included)
├── scripts/
│   ├── models/                   # Inference scripts per model
│   │   ├── qwen/                # Qwen2.5-VL (zero/few-shot)
│   │   ├── llava/               # LLaVA variants
│   │   ├── videollama/          # VideoLLaMA2
│   │   ├── gemma/               # Gemma-3
│   │   ├── gpt/                 # GPT-4o, GPT-5
│   │   └── gemini/              # Gemini 2.5 Pro
└── requirements.txt              # Dependencies
```

## Models Evaluated

**Open-source** (10 models):
- Qwen2.5-VL (72B, 32B, 7B)
- VideoLLaMA2 (72B, 7B)
- LLaVA-Video-Qwen2 (72B, 7B)
- LLaVA-Onevision-Qwen2 (72B, 7B)
- Gemma-3 (27B)

**Proprietary** (3 models):
- GPT-4o
- GPT-5
- Gemini 2.5 Pro

## Requirements

```bash
pip install -r requirements.txt
```

## Paper

Submitted for peer review.

Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
