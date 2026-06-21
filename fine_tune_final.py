"""
Fine-Tuning Project: GPT-2 on Custom Q&A Data
Portfolio Project - Successfully tested on Google Colab

This script:
1. Creates training dataset with Q&A pairs
2. Fine-tunes GPT-2 using LoRA (Parameter-Efficient Fine-Tuning)
3. Saves the fine-tuned model
4. Ready for inference

Requirements:
- torch
- transformers
- datasets
- peft
"""

import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType

print("=" * 70)
print("🚀 FINE-TUNING GPT-2 ON CUSTOM DATA")
print("=" * 70)

# ============================================================================
# STEP 1: CREATE TRAINING DATA
# ============================================================================

print("\n📝 Step 1: Creating training data...")

training_data = [
    {
        "instruction": "What is machine learning?",
        "output": "Machine learning is a type of artificial intelligence that learns from data without being explicitly programmed."
    },
    {
        "instruction": "Explain fine-tuning",
        "output": "Fine-tuning is the process of taking a pre-trained model and training it further on new, specific data to improve performance on particular tasks."
    },
    {
        "instruction": "What is Azure OpenAI?",
        "output": "Azure OpenAI is Microsoft's managed service providing access to OpenAI's powerful language models through Azure's cloud infrastructure."
    },
    {
        "instruction": "How does RAG work?",
        "output": "RAG (Retrieval-Augmented Generation) retrieves relevant documents and uses them as context when generating responses, improving accuracy."
    },
    {
        "instruction": "What is transfer learning?",
        "output": "Transfer learning is using knowledge gained from one task to improve learning on another related task, saving time and resources."
    },
    {
        "instruction": "Explain embeddings",
        "output": "Embeddings are numerical representations of text that capture semantic meaning, allowing similarity comparisons and clustering."
    },
    {
        "instruction": "What is Groq?",
        "output": "Groq is an AI infrastructure company providing fast inference using specialized LPU (Language Processing Unit) hardware."
    },
    {
        "instruction": "How do transformers work?",
        "output": "Transformers use attention mechanisms to process sequences in parallel, understanding relationships between words regardless of distance."
    },
    {
        "instruction": "What is cloud computing?",
        "output": "Cloud computing is the delivery of computing services over the internet, providing on-demand access to computing resources."
    },
    {
        "instruction": "Explain API",
        "output": "An API (Application Programming Interface) is a set of protocols allowing different software applications to communicate with each other."
    }
]

# Save training data
with open('training_data.json', 'w') as f:
    json.dump(training_data, f, indent=2)

print(f"✅ Created {len(training_data)} training examples")

# ============================================================================
# STEP 2: PREPARE DATASET
# ============================================================================

print("\n📊 Step 2: Preparing dataset...")

formatted_data = []
for example in training_data:
    text = f"Instruction: {example['instruction']}\nOutput: {example['output']}\n\n"
    formatted_data.append({"text": text})

# Save formatted data
with open("formatted_data.json", 'w') as f:
    json.dump(formatted_data, f)

# Load as HuggingFace dataset
dataset = Dataset.from_dict({"text": [d["text"] for d in formatted_data]})
print(f"✅ Dataset prepared: {len(dataset)} examples")

# ============================================================================
# STEP 3: LOAD MODEL AND TOKENIZER
# ============================================================================

print("\n🤖 Step 3: Loading model and tokenizer...")

MODEL_NAME = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
print(f"✅ Tokenizer loaded")

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
print(f"✅ Model loaded")

# ============================================================================
# STEP 4: SETUP LoRA (PARAMETER-EFFICIENT FINE-TUNING)
# ============================================================================

print("\n⚙️ Step 4: Setting up LoRA...")

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,  # Small rank for GPT-2
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none"
)

model = get_peft_model(model, peft_config)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())

print(f"✅ LoRA configured")
print(f"   Trainable params: {trainable_params:,}")
print(f"   Total params: {total_params:,}")
print(f"   Trainable %: {100 * trainable_params / total_params:.2f}%")

# ============================================================================
# STEP 5: TOKENIZE DATASET
# ============================================================================

print("\n📊 Step 5: Tokenizing dataset...")

MAX_SEQ_LENGTH = 256

def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding="max_length",
        max_length=MAX_SEQ_LENGTH,
        truncation=True
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)
print(f"✅ Dataset tokenized")

# ============================================================================
# STEP 6: FINE-TUNING
# ============================================================================

print("\n" + "=" * 70)
print("🔄 STARTING FINE-TUNING")
print("=" * 70)

training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    num_train_epochs=3,
    per_device_train_batch_size=1,  # Small batch for GPU memory
    save_steps=50,
    save_total_limit=2,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_steps=50,
    logging_steps=10,
    report_to=[],  # Disable wandb
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Train!
trainer.train()
print("\n✅ Fine-tuning complete!")

# ============================================================================
# STEP 7: SAVE MODEL
# ============================================================================

print("\n💾 Step 7: Saving model...")

model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

print("✅ Model saved to ./fine_tuned_model")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("✅ FINE-TUNING PROJECT COMPLETE!")
print("=" * 70)
print(f"\n📊 Training Summary:")
print(f"   - Model: {MODEL_NAME}")
print(f"   - Training examples: {len(training_data)}")
print(f"   - Epochs: 3")
print(f"   - Batch size: 1")
print(f"   - Learning rate: 2e-4")
print(f"   - LoRA rank: 8")
print(f"\n📁 Output:")
print(f"   - Fine-tuned model: ./fine_tuned_model/")
print(f"   - Training data: training_data.json")
print("=" * 70)
