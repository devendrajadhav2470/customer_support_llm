# Customer Support LLM — QLoRA Fine-Tuning of Gemma 3 4B

Fine-tune [Gemma 3 4B-IT](https://huggingface.co/unsloth/gemma-3-4B-it) for customer support conversations using **QLoRA** (4-bit quantization + LoRA). The model is trained on the [Bitext Customer Support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) and evaluated against the base model with ROUGE and BERTScore.

**Model on Hugging Face:** [devendrajadhav34/gemma3-bitext-support-lora](https://huggingface.co/devendrajadhav34/gemma3-bitext-support-lora)

## Overview

This project adapts a general-purpose instruction-tuned LLM into a domain-specific customer support assistant. Training is memory-efficient: the base model is loaded in 4-bit precision, and only LoRA adapter weights are updated during supervised fine-tuning (SFT).

Compared to the base Gemma 3 4B model, the fine-tuned adapter produces responses that better match the tone and structure of professional support replies.

## Results

Evaluation on 100 held-out samples from the test split:

| Metric    | Base Gemma | Fine-tuned | Change   |
|-----------|------------|------------|----------|
| ROUGE-1   | 0.342      | 0.555      | +62%     |
| ROUGE-2   | 0.082      | 0.316      | +286%    |
| ROUGE-L   | 0.189      | 0.422      | **+124%** |
| BERTScore F1 | 0.843   | 0.913      | +8%      |

The base model tends to ask many follow-up questions; the fine-tuned model responds in a more direct, support-oriented style aligned with the training data.

## Tech Stack

- **Base model:** `unsloth/gemma-3-4B-it`
- **Training:** QLoRA via [Unsloth](https://github.com/unslothai/unsloth) + [PEFT](https://github.com/huggingface/peft)
- **Framework:** PyTorch, [Hugging Face Transformers](https://huggingface.co/docs/transformers), [TRL SFTTrainer](https://huggingface.co/docs/trl)
- **Quantization:** 4-bit loading with [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- **Evaluation:** ROUGE, BERTScore
- **Demo:** Gradio

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Dataset size | 26,872 examples (95/5 train/test split) |
| LoRA rank (`r`) | 16 |
| LoRA alpha | 16 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Max sequence length | 512 |
| Epochs | 1 |
| Learning rate | 2e-4 |
| Batch size | 2 (effective 8 with gradient accumulation) |
| Optimizer | AdamW 8-bit |
| Precision | FP16 |

## Project Structure

```
fkit0/
├── README.md
└── fine-tuning-project (4).ipynb   # Full training, evaluation, and Gradio demo
```

## Getting Started

### Requirements

Install dependencies (Google Colab / Kaggle GPU recommended):

```bash
pip install "unsloth[colab-new]" datasets trl accelerate bitsandbytes peft evaluate rouge_score sacrebleu bert-score gradio
```

### Run the notebook

Open `fine-tuning-project (4).ipynb` in Colab or Kaggle with a GPU runtime. The notebook covers:

1. Model loading (4-bit) and LoRA setup
2. Dataset download and chat-template formatting
3. Supervised fine-tuning with SFTTrainer
4. Base vs. fine-tuned comparison (qualitative + ROUGE/BERTScore)
5. Saving and uploading the adapter to Hugging Face
6. Gradio interactive demo

### Load the fine-tuned model

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="devendrajadhav34/gemma3-bitext-support-lora",
    max_seq_length=512,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
```

### Example inference

```python
messages = [
    {"role": "system", "content": [{"type": "text", "text": "You are a helpful customer support assistant."}]},
    {"role": "user", "content": [{"type": "text", "text": "I accidentally placed an order and want to cancel it"}]},
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to("cuda")

outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.7, do_sample=True)
response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
print(response)
```

## Dataset

[Bitext Customer Support LLM Chatbot Training Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)

Each example includes a customer `instruction` and support `response`, formatted into a multi-turn chat with a fixed system prompt:

> *"You are a helpful customer support assistant."*

## License

Follow the licenses of the base model ([Gemma](https://huggingface.co/unsloth/gemma-3-4B-it)) and the Bitext dataset. This repository contains training code and documentation; the fine-tuned LoRA weights are hosted on Hugging Face.

## Author

Devendra Jadhav
