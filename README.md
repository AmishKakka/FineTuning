# Fine-Tuning Qwen2.5 with LoRA + Journal-Guided RL

A two-phase fine-tuning pipeline that trains a small language model to perform **chain-of-thought medical reasoning** — without any human feedback.

Used **[Qwen2.5-1.5B-Instruct](https://huggingface.co/unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit)** — a 1.5B parameter decoder-only causal language model. It has reasoning capability at small scale.


## Dataset

**[FreedomIntelligence/medical-o1-reasoning-SFT](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT)** (English split)

19,704 medical question-answer pairs with three columns:

| Column | Description |
|---|---|
| `Question` | Clinical medical question |
| `Complex_CoT` | Step-by-step chain-of-thought reasoning |
| `Response` | Final clean answer |

Split: 80% train (15,763 samples) / 20% test (3,941 samples)



## Approach


### Phase 1 — Supervised Fine-Tuning (SFT) on Question → Answer

The model first learns to produce correct medical answers before learning to reason.

**LoRA Configuration**:

```python
lora_config = FastLanguageModel.get_peft_model(
    model,
    r=6,
    lora_alpha=12,
    target_modules=["q_proj", "k_proj", "v_proj"],
    bias="none",
    use_gradient_checkpointing="unsloth"
)

Output - trainable params: 1,118,208 || all params: 1,544,832,512 || trainable%: 0.0724
```

Only **0.07% of parameters** are trained — the rest of the model stays frozen.

**Phase 1 SFT Results (1 epoch, batch size 8):**

| Metric | Value |
|---|---|
| Starting loss (Batch 0) | 1.5539 |
| Final loss (Batch 1950) | 1.3796 |
| Epoch average loss | 1.2844 |
| Training batches | 1,970 |

---

### Phase 2 — Journal-Guided Reinforcement Learning on CoT (conceptual)

After SFT, the model will learn to **reason** through a closed-loop RL process inspired by Google's [Self-Evolving Recommendation System architecture](https://arxiv.org/abs/2602.10226) (2026). No human feedback.

**The Journal** is a dynamic knowledge graph built during training (kind of similar to how we learn):

```
Nodes:  Question (with topic embedding + cluster label)
        Generated CoT (with iteration number)
        Answer (with ROUGE score vs ground truth)

Edges:  weight = 1 / (loss + 1e-4)
        high weight → model was confident → good reasoning chain
        low weight  → model struggled     → harder example
```

Every 10 batches, the journal is updated with new (Question, CoT, Answer) triplets and their confidence scores.

**Journal querying during training:** When the model encounters a new question, it:

1. Embeds the question using a lightweight encoder (MiniLM)
2. Finds the closest topic cluster via cosine similarity against cluster centroids
3. Retrieves the top-k highest-confidence reasoning chains from that cluster
4. Adds them as few-shot examples to guide CoT generation

**Every 100 iterations:** Within each topic cluster, only the highest-confidence and lowest-confidence nodes are kept. This keeps the graph compact and ensures the model always sees both highest scroed examples and its failure cases.


## Project Structure

```
FineTuning/
├── model.py          # Unsloth model + LoRA
├── tokenization.py   # Tokenizer loading
├── data.py           # Dataset loading, SFT + RL tokenization, DataLoader
├── trainer.py        # SFTTrainer + RLTrainer
├── rewards.py        # ROUGE + structure reward functions
└── README.md
```

Used Google Colab's free tier Tesla T4 GPU (15.64 GB VRAM)

## Pointers
1. Phase 2 is just what i thought can be done, as an aim to develop knowledge while training just like how we learn.

2. Tried to train on Macbook but ran out of storage pretty quickly.

3. Using **peft** and **torch** libraries together gave many import errors.