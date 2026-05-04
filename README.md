# Karma-MV: A Benchmark for Causal Question Answering on Music Videos

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/TODO)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-KARMA--MV-yellow)](https://huggingface.co/datasets/amaai-lab/Karma-MV)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](https://www.apache.org/licenses/LICENSE-2.0)

> **Karma-MV** is a large-scale multiple-choice QA benchmark for causal audio-visual reasoning in music videos — testing how well models understand the relationship between visual dynamics and musical structure.

---

## Overview

While significant progress has been made in video question answering and cross-modal understanding, causal reasoning about how visual dynamics drive musical structure in music videos remains under-explored. Karma-MV addresses this with 37,737 MCQs derived from 2,682 YouTube music videos, spanning three reasoning types:

- **Evidence Reasoning** — why did the music change given the visual transition?
- **Predictive** — how will the music change given an upcoming visual change?
- **Counterfactual** — how would the music differ under an alternative visual scenario?

MCQs were generated and validated using the Qwen-2.5-7B-Instruct LLM. Each question includes an explanation of the correct answer.

---

## Repository Structure

```
Karma-MV/
├── causal_knowledge_graph/   # Construction and querying of the Causal Knowledge Graph (CKG)
├── mcq_inference/            # MCQ answering pipelines
│   ├── llm/                  # LLM-based inference (text-only)
│   └── vlm/                  # VLM-based inference (vision + language)
├── data/                     # Sample JSON files (scene-transition pairs + MCQs)
├── evaluation/               # Evaluation scripts and metrics
└── README.md
```

### Causal Knowledge Graph (CKG)

The CKG encodes structured cross-modal dependencies between visual and musical features extracted from music videos. It is used at inference time to retrieve relevant causal context and augment model inputs, improving performance — especially for smaller models.

### MCQ Inference

Two inference pipelines are provided:

- **LLM** — text-only inference using a language model, optionally augmented with CKG retrieval
- **VLM** — vision-language model inference that takes scene clip pairs as visual input, optionally augmented with CKG retrieval

Both pipelines follow the same CKG augmentation interface, making it straightforward to ablate with and without graph grounding.

---

## Dataset

The full dataset is available on HuggingFace:
👉 [https://huggingface.co/datasets/amaai-lab/Karma-MV](https://huggingface.co/datasets/amaai-lab/Karma-MV)

Each JSON file corresponds to one music video and contains a list of scene-transition pair objects:

```json
{
  "current_scene": {
    "name": "scene_003.mp4",
    "start_time": "00:00:13.833",
    "end_time": "00:00:16.542"
  },
  "past_scene": {
    "name": "scene_002.mp4",
    "start_time": "00:00:10.125",
    "end_time": "00:00:13.833"
  },
  "questions": [
    {
      "type": "Evidence Reasoning",
      "question": "...",
      "options": { "a": "...", "b": "...", "c": "...", "d": "..." },
      "answer": "a",
      "explanation": "..."
    }
  ]
}
```

---

## Getting Started

```bash
git clone https://github.com/AMAAI-Lab/Karma-MV.git
cd Karma-MV
pip install -r requirements.txt
```

TODO:
1. The MCQ files for each video are named as -- YouTubeID_qa.json.
2. Yo need to execute the following scripts:
   a. qwen_omni.py -- This script deals with Qwen-2.5-omni VLM
   b. mini_cpm-o-4_5.py -- This script deals with 

---

## Citation

If you use Karma-MV in your research, please cite:

```bibtex
@article{ghosh2026karmamv,
  author    = {Archishman Ghosh and Abhinaba Roy and Dorien Herremans},
  title     = {{Karma-MV}: A Benchmark for Causal Question Answering on Music Videos},
  year      = {2026},
  journal   = {arXiv preprint}
}
```

---

## License

This project is licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).

## Contact

For questions, please open a GitHub issue or contact the authors via the [AMAAI Lab](https://amaailab.com).
