# Multimodal Food Image Classification with LLM

Reproduction-oriented project for:

**Multimodal Food Image Classification with Large Language Models**

## Pipeline

1. Image → CLIP image encoder
2. Image → BLIP2 caption generator
3. Captions → CLIP text encoder
4. Image/Text features → self-attention + cross-attention fusion
5. Classifier → Food-101 label

## Project structure

```text
multimodal-food-classification
├── README.md
├── requirements.txt
├── configs
│   └── config.yaml
├── data
├── datasets
│   └── food101_dataset.py
├── captioning
│   └── generate_captions.py
├── embeddings
│   ├── clip_image_encoder.py
│   └── clip_text_encoder.py
├── models
│   └── multimodal_model.py
├── training
│   └── train.py
├── evaluation
│   └── evaluate.py
└── utils
    └── seed.py
```

## Setup

```bash
pip install -r requirements.txt
```

## Configuration

Edit `configs/config.yaml` for training hyperparameters and device.

## Run

Train:

```bash
python training/train.py
```

Evaluate (after training checkpoint is created):

```bash
python evaluation/evaluate.py
```

## Notes on compute

- BLIP2 is the heaviest component and benefits from a large GPU.
- Training in this baseline generates captions online, so it is slow.
- For production-like speed, precompute captions/features and cache to disk.
