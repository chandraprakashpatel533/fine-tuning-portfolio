# Fine-Tuning Project: GPT-2 on Custom Q&A Data

## Overview
Fine-tuned GPT-2 model on custom Q&A dataset using LoRA (Parameter-Efficient Fine-Tuning).

## Quick Start

### Install
```bash
pip install -r requirements.txt
```

### Run
```bash
python fine_tune_final.py
```

### Output
- Fine-tuned model: `fine_tuned_model/`
- Training data: `training_data.json`
- Takes ~10-15 minutes on GPU (5+ minutes on CPU)

## What's Included
- ✅ Complete fine-tuning pipeline
- ✅ 10 Q&A training examples
- ✅ LoRA for efficient training
- ✅ Automatic model saving

## Technology
- Model: GPT-2
- Method: LoRA (Low-Rank Adaptation)
- Training: HuggingFace Transformers
- Platform: Google Colab (tested and working)

## Results
- Training loss decreased over 3 epochs
- Model successfully fine-tuned on custom data
- Ready for inference on domain-specific Q&A

## Portfolio Value
Shows understanding of:
✅ Model fine-tuning
✅ Parameter-efficient training
✅ Data preparation
✅ ML model deployment