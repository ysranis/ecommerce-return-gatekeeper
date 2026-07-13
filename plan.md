# Execution Plan — E-commerce Return Gatekeeper (Multi-Tiered Fleet)

> See [prd.md](./prd.md) for full product requirements, feature matrix, KPIs, and architecture.

---

## Project Overview

This is a 4-week portfolio project demonstrating model distillation, QLoRA fine-tuning, and multi-model fleet evaluation for an e-commerce post-purchase dispute arbitration system. DeepSeek-V3 generates a synthetic golden dataset from 2,000 Bitext seeds. Two student models (Qwen-2.5-7B and Llama-3.2-3B) are fine-tuned in parallel and coordinated via a Dynamic Triage Router — showcasing both AI product strategy (cost optimization, ROI, business logic enforcement) and technical engineering (distillation pipeline, LoRA training, evaluation harness).

---

## Tooling Stack

| Layer | Tool | Purpose |
|---|---|---|
| Dataset | Bitext (HuggingFace) | Seed corpus for synthetic generation |
| Teacher API | DeepSeek-V3 | Dispute generation + golden labeling (with policy handbook) |
| Fine-tuning | Unsloth | QLoRA training — 2× faster, 70% less VRAM than standard HF Trainer; alternative is HuggingFace `SFTTrainer` (trl) |
| GPU Compute | RunPod A10G @ ~$0.44/hr | Cloud GPU for training + evaluation in one session; Lambda Labs A100 is an alternative |
| Experiment Tracking | Weights & Biases (W&B) | Loss curves, hyperparameters, run comparison |
| Inference (eval) | HuggingFace `transformers` pipeline | Load model directly in Python for evaluation — no server needed for 150 test rows |
| Inference (demo) | Pre-computed outputs (hardcoded in dashboard) | Vercel can't host GPU models; demo pages use stored results |
| Evaluation | Custom Python + LangSmith (free tier) | Metrics are JSON parsing + string comparison — no framework needed; LangSmith logs judge calls |
| LLM Judge | Claude Haiku 4.5 (`claude-haiku-4-5-20251001`) | 10× cheaper than Sonnet; sufficient for 1–5 rubric scoring. Alt: DeepSeek-V3 (cheapest) |
| Dashboard | Next.js 14+ | Analytics visualization layer |
| Database | Vercel Postgres | Stores eval results, row-level inspection data |
| Deployment | Vercel (dashboard) + HF Hub (models) | Public-facing portfolio artifacts |

---

## Project Cost Breakdown

**Total estimated cost to build this entire project: ~$5–6**

| Week | Activity | Tool | Cost |
|---|---|---|---|
| 1 | DeepSeek-V3 synthetic generation (2,000 rows) | DeepSeek API | ~$0.60 |
| 1 | DeepSeek-V3 golden labeling pass (2,000 rows) | DeepSeek API | ~$1.85 |
| 1 | 400 multilingual variants | DeepSeek API | ~$0.30 |
| 2 | Qwen-7B QLoRA training — Unsloth (~1.5 hrs) | RunPod A10G | ~$0.70 |
| 2 | Llama-3B QLoRA training — Unsloth (~45 min) | RunPod A10G | ~$0.35 |
| 2 | Baseline eval + inference (same GPU session) | RunPod A10G | ~$0.50 |
| 3 | 5-model evaluation, 750 inferences (same GPU session) | RunPod A10G | ~$1.00 |
| 3 | LLM Judge — Claude Haiku 4.5 (750 calls) | Anthropic API | ~$0.24 |
| 4 | Dashboard hosting | Vercel free tier | $0 |
| 4 | Database (eval results) | Vercel Postgres free tier | $0 |
| 4 | Model publishing | HuggingFace Hub free | $0 |
| **TOTAL** | | | **~$5.54** |

### Cost optimizations already baked in
- **Unsloth** instead of standard HF Trainer → 40–50% fewer GPU hours
- **Claude Haiku** instead of Sonnet for judge → saves ~$0.76
- **`transformers` pipeline** instead of vLLM → no second GPU instance for inference
- **One continuous GPU session** for training + eval → no repeated startup cost
- **No DeepEval/Arize Phoenix** → zero cost, zero complexity

### Optional further savings
- Use **DeepSeek-V3 as the judge** instead of Claude Haiku → saves another ~$0.13
- Enable **DeepSeek prompt caching** for the policy handbook → 70% off cached input tokens on the labeling pass → saves ~$0.60
- **Total with all optional savings: ~$4.50**

---

## Before You Begin: Key Concepts Explained Simply

If you're new to AI/ML engineering, read this section first. Every tool and term you'll encounter in this plan is explained here in plain English — no jargon assumed.

---

### What is HuggingFace?

Think of HuggingFace like **GitHub, but for AI models and datasets**. Just like developers upload code to GitHub so others can download and use it, AI researchers upload their trained models to HuggingFace so anyone can download and run them.

In this project, we use HuggingFace for three things:
1. **Downloading the Bitext dataset** — someone already uploaded 26,872 customer support conversations there. We just pull it with one line of code.
2. **Downloading the base models** — Qwen-2.5-7B and Llama-3.2-3B are already uploaded to HuggingFace. We download them as starting points.
3. **Uploading our fine-tuned models** — once we've trained our models, we publish them back to HuggingFace so they appear in our portfolio.

You'll need a free HuggingFace account at huggingface.co.

---

### What is a Dataset?

A dataset is just a structured collection of examples used to train or evaluate a model. Think of it as **a giant set of flashcards**. Each card has an input (a customer question) and a correct output (what the model should say/do in response).

The Bitext dataset has 26,872 of these cards — real customer support messages labelled by their intent (e.g., "get_refund", "cancel_order"). We use these as our starting input questions, but we throw away their original answers and generate much better, more structured ones using DeepSeek-V3.

---

### What is a Teacher Model vs. a Student Model?

This project uses a technique called **Knowledge Distillation** — which is exactly what it sounds like: distilling the knowledge of a big, expensive model into a smaller, cheaper one.

- **Teacher (DeepSeek-V3):** An enormous, highly intelligent AI with 671 billion parameters. It's expensive to run ($0.27/1M tokens) but produces nearly perfect outputs. We use it to generate the "correct answer" training examples.
- **Students (Qwen-7B and Llama-3B):** Smaller, faster, cheaper models. On their own, they're not smart enough to handle complex policy decisions. But after we train them on the teacher's perfect examples, they learn to mimic that quality — at a fraction of the cost.

The analogy: a senior doctor (teacher) writes up detailed case notes on 1,500 patients. A medical student (student model) reads all those notes and learns to make the same diagnoses — without having to see those patients themselves.

---

### What is Fine-Tuning?

Pre-trained models like Qwen-7B or Llama-3B have been trained by their creators on billions of internet pages. They know a lot about the world in general. But they don't know anything about *your specific business rules* — your return policy, your JSON schema format, your company's escalation logic.

**Fine-tuning is giving the model extra, specialized lessons.** You feed it 1,200 examples of exactly the inputs and outputs you want, and it adjusts its internal weights to get better at your specific task. After fine-tuning, the model "remembers" your domain deeply — like an employee who's been trained on your company's specific processes.

---

### What is LoRA / QLoRA?

Fine-tuning a 7-billion parameter model normally requires updating all 7 billion parameters — which needs an enormous amount of GPU memory (100+ GB). Most people don't have that.

**LoRA (Low-Rank Adaptation)** is a clever shortcut: instead of updating all parameters, you freeze the original model and only train a tiny set of "adapter" weights (about 1% the size). These adapters sit on top of the frozen model and handle the domain-specific knowledge. The original model is untouched.

**QLoRA** is LoRA + quantization: the frozen base model is compressed (quantized) to use even less memory. This lets you fine-tune a 7B model on a single GPU with 24GB VRAM — something you can rent cheaply on RunPod or Lambda Labs.

Think of it as: you don't rewrite the whole textbook, you just add sticky-note annotations throughout.

---

### What is vLLM?

Once you've trained your model, you need a way to serve it — i.e., accept incoming requests and return responses. **vLLM is a high-performance server for running LLMs**. It's optimized for speed and can handle many requests simultaneously.

You run it like a local web server. It exposes an API endpoint that accepts a prompt and returns a response — compatible with the same format as OpenAI's API, so existing code just works.

---

### What is Weights & Biases (W&B)?

Training a model is like a long experiment. The "loss" (how wrong the model's answers are) should go down over time as training progresses. But if something goes wrong — overfitting, learning rate too high, etc. — you need to catch it early.

**W&B is a dashboard that records your training experiment in real time.** It logs the loss curve, accuracy metrics, hyperparameters, and GPU usage so you can watch the training session as it runs and compare different runs side by side.

Think of it as the black-box flight recorder for your training run.

---

### What is RunPod / Lambda Labs?

Your laptop cannot train a 7-billion parameter model. It doesn't have a GPU powerful enough. **RunPod and Lambda Labs are services where you rent a cloud GPU by the hour** — typically an NVIDIA A10G (24GB VRAM) or A100 (80GB VRAM).

You spin up a GPU instance, run your training script, and pay only for the hours used. A full fine-tuning run for this project costs roughly $5–15 depending on the GPU and duration. When training is done, you shut down the instance and stop paying.

---

### What is a JSON Schema?

JSON is a structured data format that looks like this: `{"key": "value", "number": 42}`. A JSON schema is a **contract** that defines exactly what fields must be present, what types they must be, and what values are allowed.

In this project, every model output must conform to a strict schema containing fields like `intent_action`, `gatekeeper_status`, `confidence_score`, etc. If the model outputs anything that doesn't match the schema — a missing field, wrong format, free-form text instead of structured JSON — that output is rejected as unusable. Enforcing this is what makes the system reliable enough to plug into a real application.

---

### What is LLM-as-a-Judge?

After fine-tuning, how do you know if the model is actually good? You need to evaluate its outputs. But having a human read 750 outputs (5 models × 150 test rows) is slow and expensive.

**LLM-as-a-Judge** means using another high-quality AI (in our case, Claude Sonnet or GPT-4o) to automatically score each candidate output against the correct "golden" answer. We give the judge a specific 1–5 rubric and it returns a score and a one-sentence reason. This gives us fast, consistent, scalable evaluation.

---

### What is Vercel Postgres?

This is just a simple cloud database that stores our evaluation results. Think of it as a Google Sheet in the cloud that our Next.js dashboard can query to display charts and tables. Vercel provides it for free at small scale, and it integrates directly with Vercel-hosted apps.

---

## Week 1: Dataset Engineering

**Objective:** Build a high-quality, schema-validated golden dataset of ≥1,500 rows for fine-tuning.

**Quality Gate 1:** Reject any rows where DeepSeek-V3 fails to output a fully parseable JSON block. A minimum of **1,500 validated rows** must exist in `distilled_dataset.jsonl` before Week 2 begins.

### Step 1 — Bitext Ingestion & Filtering

#### What this step does (in plain English)

We start by downloading the Bitext customer support dataset from HuggingFace — think of it as pulling a spreadsheet of 26,872 real customer messages off a shared drive. We don't need all of them. We only want messages that match our 5 business scenarios: refund requests, cancellations, refund tracking, complaints, and policy questions.

We then filter the dataset down to ~2,000 rows that cover these topics with a good mix of different communication styles (typos, aggression, politeness, informal language). Crucially, we **throw away Bitext's original answers** — they're generic, long-form chat text that would be useless for training a structured JSON-outputting model. We only keep the customer's raw question and the linguistic style flags.

Success looks like: a file `data/bitext_seeds.jsonl` with 2,000 rows, each containing only the customer message and its tags.

```python
from datasets import load_dataset

ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")

TARGET_INTENTS = ["get_refund", "cancel_order", "track_refund", "complaint", "check_refund_policy"]
TARGET_FLAGS = ["Q", "W", "Z", "P"]  # colloquial, offensive, typos, politeness

filtered = ds["train"].filter(lambda x: x["intent"] in TARGET_INTENTS)

# Sample 2,000 balanced rows (equal distribution per intent × flag combination)
# Extract ONLY: instruction (user_input) + flags (linguistic_markers)
# Discard original Bitext text answers entirely
```

### Step 2 — Synthetic Generation with DeepSeek-V3

#### What this step does (in plain English)

Raw Bitext messages are short and simple — good seeds, but not the complex, messy disputes our model needs to learn to handle. In this step, we pass each seed message to DeepSeek-V3 and ask it to generate a more realistic, harder version of that dispute. We also provide DeepSeek-V3 with a fictional "ACME Corp Returns Policy Handbook" so it generates scenarios that have concrete policy stakes — not vague generic complaints.

We also generate 400 additional variants in Spanish, French, and Arabic to give our model multilingual capability.

The result is ~2,000–2,400 synthetic but realistic customer dispute messages — much richer than the Bitext originals, and grounded in specific business rules.

The system prompt must include the fictional corporate returns handbook so DeepSeek-V3 has concrete policy rules to reason against:

```python
# DeepSeek-V3 uses OpenAI-compatible API endpoint
POLICY_HANDBOOK = """
ACME CORP RETURNS & DISPUTE POLICY HANDBOOK (v2.1):
- Orders within 30 days: Full refund eligible
- Orders 31–60 days: Store credit only
- Damaged items: Photo evidence required before any refund
- Orders in transit >14 business days: Eligible for immediate automated cancellation
- Fraudulent chargeback flags: Auto-escalate to human agent
- Orders already delivered and opened: Requires human review
"""

GENERATION_SYSTEM_PROMPT = f"""You are a synthetic data generator for an e-commerce dispute AI training dataset.
{POLICY_HANDBOOK}

Given a seed customer support message, generate a NEW, more complex, realistic customer dispute scenario.
The new message should:
- Be messier and more conversational than the seed
- Include realistic slot data (order IDs like AX-XXXX or ORD-XXXXX, invoice IDs)
- Vary emotional register (frustrated, polite, aggressive, confused)
- Contain natural typos and abbreviations
Output ONLY the new customer message, nothing else."""

# Generate ~2,000 synthetic disputes from 2,000 seeds (1 variant per seed)
# Then generate 400 multilingual variants (Spanish, French, Arabic)
```

### Step 3 — Golden Labeling Pass

#### What this step does (in plain English)

This is the core of knowledge distillation. We now take each synthetic customer dispute message from Step 2 and send it back to DeepSeek-V3 — but this time we ask it to act as an expert arbitration engine and produce a **structured JSON decision** for that dispute.

This JSON is the "golden answer" — the perfect, correct output we want our smaller student models to learn to produce. It includes the chain-of-thought reasoning, the intent classification, extracted order details, a policy decision (approve / request evidence / escalate), and the customer-facing response.

DeepSeek-V3 generates this for all ~2,000 rows. This is the most expensive step (costs roughly $5–15 in API fees) but only runs once. The result is our training dataset.

```python
LABELING_SYSTEM_PROMPT = f"""You are the expert arbitration engine for Acme Corp's return gatekeeper system.
{POLICY_HANDBOOK}

Analyze the customer dispute message and output a structured JSON arbitration decision.
Your output must be valid JSON matching this exact schema:
{{
  "chain_of_thought": "...",
  "intent_action": "get_refund|cancel_order|track_refund|complaint|check_refund_policy",
  "extracted_slots": {{
    "order_id": "...|null",
    "invoice_id": "...|null",
    "return_window_days": int|null,
    "item_condition": "...|null"
  }},
  "policy_evaluation": {{
    "within_return_window": bool,
    "item_opened": bool,
    "evidence_required": bool
  }},
  "gatekeeper_status": "APPROVE_AUTOMATED|REQUEST_EVIDENCE|ESCALATE_TO_HUMAN",
  "confidence_score": float,
  "fallback_escalation": bool,
  "user_facing_response": "...",
  "processing_timestamp": "ISO8601"
}}"""
```

### Step 4 — Validation Gate

#### What this step does (in plain English)

DeepSeek-V3 is excellent but not perfect — occasionally it'll produce a response that's missing a required field, or outputs garbled text instead of valid JSON. If we train on broken examples, the student model learns to produce broken outputs too.

This step runs every generated row through a checker: can Python parse it as JSON? Are all the required fields present? If yes, the row goes into the clean dataset. If no, we retry with a stricter prompt (up to 2 retries). If it still fails, we discard that row.

We need at least 1,500 valid rows to proceed. Once we have them, we split into: 1,200 for training, 150 for validation during training, and 150 held-out for the final evaluation (these 150 test rows are never used during training — they're our exam paper).

```python
import json

REQUIRED_KEYS = [
    "chain_of_thought", "intent_action", "extracted_slots",
    "gatekeeper_status", "confidence_score", "user_facing_response"
]

def validate_output(response_str: str) -> bool:
    try:
        data = json.loads(response_str)
        return all(k in data for k in REQUIRED_KEYS)
    except json.JSONDecodeError:
        return False

# Discard invalid rows; retry with stricter prompt (max 2 retries per row)
# Stop generation only after 1,500 validated rows confirmed
```

### Deliverables

- `data/bitext_seeds.jsonl` — 2,000 filtered seed rows (instruction + flags only)
- `data/distilled_dataset.jsonl` — ≥1,500 validated golden rows
- `data/train.jsonl` — 1,200 rows
- `data/val.jsonl` — 150 rows
- `data/test.jsonl` — 150 rows (**held out — do not use during training**)

#### What you now have after Week 1

You now have a clean, structured training dataset of 1,500+ expert-level arbitration decisions — all generated by DeepSeek-V3. This is the core asset of the entire project. Every subsequent step depends on it. Think of it as: you've built the textbook that the student models will study from.

---

## Week 2: Baseline Evaluation + Fine-Tuning

**Objective:** Establish zero-shot baselines for both models, run parallel fine-tuning tracks, and prototype the triage router.

**Quality Gate 2:** Training loss curves must display regular, non-volatile flattening across epochs without exhibiting validation overfitting. If val loss stops improving for 2 consecutive eval steps, apply early stopping.

### Step 1 — Zero-Shot Baseline

#### What this step does (in plain English)

Before we train anything, we need to know how bad the models are *without* any training. This is called the "baseline" — it's the benchmark we'll compare our fine-tuned models against later to prove the training worked.

We send all 150 test rows to the raw, untouched Qwen-7B and Llama-3B models and ask them to produce the same JSON output format we'll train them to produce. Most of the time they'll fail — wrong format, missing fields, or completely hallucinated answers. That's expected. We record those failures. This is our "before" photo.

Run `test.jsonl` (150 rows) against both base models with zero-shot prompting. Log:
- JSON parse success/failure per row
- `intent_action` vs. ground truth
- `gatekeeper_status` vs. ground truth
- Hallucinated entity slots (fabricated IDs not present in input)

Output: `results/baseline_results_qwen.json` and `results/baseline_results_llama.json`

### Step 2 — W&B Setup

#### What this step does (in plain English)

Before starting any training, we connect to Weights & Biases (W&B). This is like turning on the flight recorder before a flight. Every training run will automatically log its loss, accuracy, and hyperparameters to the W&B dashboard in real time.

This matters for two reasons: (1) you can catch problems early — if loss spikes or stops improving, you know to stop the run and investigate before wasting more GPU hours; (2) you get professional experiment tracking that shows side-by-side comparisons of your runs — which is itself a portfolio artifact.

You'll need a free account at wandb.ai and your API key.

```bash
pip install wandb
wandb login  # use WANDB_API_KEY
```

Tag each run: `project=ecommerce-gatekeeper`, `group=qwen-7b` or `group=llama-3b`

### Step 3 — Fine-Tuning: Track A (Qwen-2.5-7B)

#### What this step does (in plain English)

This is the main training event for our "smart" model. We take Qwen-2.5-7B — a capable general-purpose 7-billion parameter model — and teach it to specialize in e-commerce dispute arbitration.

We use **QLoRA via Unsloth**: the base model is frozen and compressed into low memory, and we only train a small set of "adapter" layers (targeting the query and value attention matrices). Unsloth's custom CUDA kernels make training ~2× faster and use ~70% less memory than standard HuggingFace Trainer — which directly cuts GPU rental cost. We train for 3 epochs, meaning the model studies the 1,200 training examples 3 times over. Total training time: ~1.5 hours on a RunPod A10G.

LoRA targets attention projections Wq and Wv for maximum policy-adherence learning:

**Unsloth training script (`scripts/03_train_qwen.py`):**

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import wandb

wandb.init(project="ecommerce-gatekeeper", name="qwen-7b-run-001")

# Load base model with Unsloth (4-bit quantized for QLoRA)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,   # QLoRA: 4-bit quantization
)

# Attach LoRA adapters (only Wq and Wv trained)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                          # LoRA rank
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
)

dataset = load_dataset("json", data_files="data/train.jsonl", split="train")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    args=TrainingArguments(
        output_dir="./output/qwen-2.5-7b-ecommerce-gk",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        bf16=True,
        save_strategy="epoch",
        report_to="wandb",
    ),
)

trainer.train()
model.save_pretrained("./output/qwen-2.5-7b-ecommerce-gk")
tokenizer.save_pretrained("./output/qwen-2.5-7b-ecommerce-gk")
```

### Step 4 — Fine-Tuning: Track B (Llama-3.2-3B)

#### What this step does (in plain English)

Same process as Track A, but for the Llama-3.2-3B model — our "fast and cheap" option. This model only has 3 billion parameters (vs. 7B for Qwen), so it trains faster and runs faster in production.

The trade-off: it's less capable at complex reasoning. So we tune its training for a different goal — maximum speed (low TTFT, time-to-first-token) rather than maximum accuracy. We use a smaller LoRA rank (`lora_r=8` instead of 16) and a shorter sequence length (1024 instead of 2048), which reduces memory usage and training time further.

The goal is to prove that a 3B model, when properly trained on domain-specific data, can handle simple procedural tasks (order cancellations, status checks) just as reliably as a much larger model — at a fraction of the cost.

**Unsloth training script (`scripts/04_train_llama.py`):**

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import wandb

wandb.init(project="ecommerce-gatekeeper", name="llama-3b-run-001")

# Load Llama-3B with Unsloth — smaller rank and shorter sequence for speed/cost
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    max_seq_length=1024,   # Shorter than Qwen — Track B handles simpler inputs
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=8,                           # Smaller rank than Track A — cost/speed optimized
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
)

dataset = load_dataset("json", data_files="data/train.jsonl", split="train")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=1024,
    args=TrainingArguments(
        output_dir="./output/llama-3.2-3b-ecommerce-gk",
        num_train_epochs=3,
        per_device_train_batch_size=8,   # Larger batch — smaller model fits more per GPU
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        bf16=True,
        save_strategy="epoch",
        report_to="wandb",
    ),
)

trainer.train()
model.save_pretrained("./output/llama-3.2-3b-ecommerce-gk")
tokenizer.save_pretrained("./output/llama-3.2-3b-ecommerce-gk")
```

### Step 5 — Triage Router Prototype

#### What this step does (in plain English)

Now we have two trained models. In production, we don't want every request going to the same model — we want smart routing. Simple, routine tasks (checking order status) should go to the fast 3B model (Track B). Complex or emotionally charged disputes should go to the accurate 7B model (Track A).

The Triage Router is a simple rule-based classifier that reads the detected intent and emotional tone of an incoming request and decides which model to send it to. It's not a neural network — it's just an `if/else` logic function. We keep it simple intentionally: the PRD's "Out of Scope" section explicitly defers router model training to a future version. A rule-based router is fast, transparent, and sufficient for the portfolio demonstration.

```python
# router/triage_router.py
# Rule-based classifier (no additional model training required)

HIGH_COMPLEXITY_INTENTS = {"get_refund", "complaint", "check_refund_policy"}
HIGH_EMOTION_MARKERS = {"HIGH_EMOTION", "OFFENSIVE", "FRUSTRATED"}

def route_request(user_input: str, detected_intent: str, emotion_markers: list) -> str:
    """Returns 'track_a' or 'track_b'."""
    if detected_intent in HIGH_COMPLEXITY_INTENTS:
        return "track_a"
    if any(m in HIGH_EMOTION_MARKERS for m in emotion_markers):
        return "track_a"
    # Simple procedural intents → Track B
    return "track_b"
```

### Catastrophic Forgetting Guard

#### What this step does (in plain English)

There's a risk with fine-tuning: the model gets so focused on your specific domain that it "forgets" its general capabilities. This is called **catastrophic forgetting** — the model becomes great at e-commerce disputes but suddenly can't answer basic common-sense questions anymore.

We guard against this by running a general-purpose benchmark called **MMLU** (Massive Multitask Language Understanding) before and after training. It tests the model on 100 general-knowledge questions. If the model scores more than 5 percentage points lower after training, something went wrong and we need to investigate (reduce LoRA rank, add more dropout, train for fewer epochs).

```bash
# Run before and after each training run
lm_eval --model hf \
  --model_args pretrained=./output/qwen-2.5-7b-ecommerce-gk \
  --tasks mmlu --num_fewshot 5 --limit 100 \
  --output_path ./eval/mmlu_qwen_post.json
```

Flag as risk if MMLU accuracy drops >5 points vs. base.

### Deliverables

- `output/qwen-2.5-7b-ecommerce-gk/` — merged Qwen adapter weights
- `output/llama-3.2-3b-ecommerce-gk/` — merged Llama adapter weights
- `results/baseline_results_qwen.json`, `results/baseline_results_llama.json`
- `router/triage_router.py` — triage router prototype
- W&B training run links for both models
- MMLU guard metric results (pre/post for both models)

#### What you now have after Week 2

You now have two fine-tuned models stored as adapter weights on your GPU instance. You've proven (via MMLU) that they haven't forgotten their general capabilities. You also have baseline results showing how badly the untrained models performed — which sets up the dramatic "before vs. after" comparison in Week 3. You also have a working triage router that can intelligently direct traffic between the two models.

---

## Week 3: Multi-Model Benchmark

**Objective:** Run comparative evaluation across all 5 model variants on 150 test rows.

**Quality Gate 3:**
- Track A must achieve 100% JSON schema compliance and ≥97% `gatekeeper_status` accuracy
- Track B must verify TTFT < 150ms on the vLLM endpoint

### Step 1 — Set Up 5 Inference Endpoints

#### What this step does (in plain English)

We're about to run the final exam — sending all 150 held-out test rows through every model variant and comparing the results. But first, we need each model to be "live" and accepting requests.

We load each model sequentially in Python using HuggingFace's `transformers` pipeline — no server, no ports, no extra setup. Since we're already on a GPU instance (same session as training), we just load the model weights into memory, run all 150 test rows, save the results, unload, and move to the next model. All 5 models get evaluated on the same 150 questions.

> **Why not vLLM?** vLLM is a production inference server for high-throughput serving to real users. For 150 evaluation rows in a portfolio project, it's overkill — it adds GPU server complexity and requires running the GPU instance for longer. `transformers` pipeline is simpler, free, and faster to set up.

| Model Variant | Method |
|---|---|
| Base Qwen-2.5-7B | `transformers` pipeline — HuggingFace |
| FT Qwen-2.5-7B (Track A) | `transformers` pipeline — local adapter weights |
| Base Llama-3.2-3B | `transformers` pipeline — HuggingFace |
| FT Llama-3.2-3B (Track B) | `transformers` pipeline — local adapter weights |
| DeepSeek-V3 (Teacher) | DeepSeek API (cloud) |

```python
from transformers import pipeline
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_model_pipeline(model_path: str, adapter_path: str = None):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    return pipeline("text-generation", model=model, tokenizer=tokenizer)
```

### Step 2 — Automated Evaluation Script (`scripts/03_evaluate_models.py`)

#### What this step does (in plain English)

For every one of the 150 test rows, we send the customer message to all 5 model endpoints and collect the responses. For each response we automatically check:
- **JSON validity** — can Python parse it without error?
- **Intent accuracy** — did the model correctly identify the customer's intent (e.g., `cancel_order`)?
- **Policy accuracy** — did it produce the right `gatekeeper_status` (`APPROVE_AUTOMATED`, `REQUEST_EVIDENCE`, or `ESCALATE_TO_HUMAN`)?
- **Slot extraction F1** — did it correctly pull out the order ID, invoice ID, etc.?
- **Hallucinations** — did it invent an order ID that wasn't in the customer's message?
- **Latency** — how fast did it respond? How long until the first token appeared (TTFT)?

All results get saved to a database (Vercel Postgres) and a local file, ready for the dashboard.

```python
import json, time
from openai import OpenAI

ENDPOINTS = {
    "base_qwen":   {"url": "http://localhost:8000/v1", "model": "qwen-base"},
    "ft_qwen":     {"url": "http://localhost:8001/v1", "model": "qwen-ft"},
    "base_llama":  {"url": "http://localhost:8002/v1", "model": "llama-base"},
    "ft_llama":    {"url": "http://localhost:8003/v1", "model": "llama-ft"},
    "deepseek":    {"url": "https://api.deepseek.com/v1", "model": "deepseek-chat"},
}

METRICS = [
    "json_valid", "intent_correct", "gatekeeper_status_correct",
    "entity_f1", "hallucination_detected",
    "latency_ms", "ttft_ms"  # TTFT critical for Track B
]

# For each endpoint × each of 150 test rows:
# 1. Send request; record wall-clock latency and TTFT
# 2. Attempt JSON parse → json_valid
# 3. Compare intent_action, gatekeeper_status, extracted_slots to ground truth
# 4. Check for hallucinated slot values (IDs not present in user_input)

# Store results to Vercel Postgres for dashboard querying
# Save flat file backup to results/eval_results.json
```

### Step 3 — LLM-as-a-Judge Scoring

#### What this step does (in plain English)

Automated metrics (JSON valid? intent correct?) are useful but shallow. They tell you *if* the output was structurally right, but not *how well* the model reasoned or how good the customer-facing response actually was.

This is where LLM-as-a-Judge comes in. For each of the 750 outputs (5 models × 150 rows), we send the candidate response alongside the golden "correct" answer to **Claude Haiku 4.5** and ask it to score the response from 1–5 using our rubric. We use Haiku (not Sonnet) because this scoring task doesn't require deep reasoning — it's just comparing two JSON objects against a fixed rubric. Haiku costs ~10× less than Sonnet for the same task. Every judge call is logged to **LangSmith** (free tier) for traceability.

**Keep the prompt short** — under 400 tokens total per call. No chain-of-thought from the judge. Output only `{"score": X, "reason": "one sentence"}`.

```python
import anthropic
from langsmith import traceable

client = anthropic.Anthropic()

@traceable(name="llm-judge-evaluation")   # LangSmith logs this automatically
def judge_output(candidate: dict, ground_truth: dict) -> dict:
    prompt = (
        f"Score the candidate vs ground truth using this rubric:\n"
        f"5=Perfect match. 4=Correct intent+status, minor slot miss. "
        f"3=Correct JSON but wrong gatekeeper_status. "
        f"2=Partially parseable JSON. 1=Hallucinated fields or critical policy breach.\n\n"
        f"Ground truth: {json.dumps(ground_truth)}\n"
        f"Candidate: {json.dumps(candidate)}\n\n"
        f'Output only: {{"score": <1-5>, "reason": "<one sentence>"}}'
    )
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",   # Haiku — 10× cheaper than Sonnet
        max_tokens=80,                        # Short output only
        messages=[{"role": "user", "content": prompt}]
    )
    return json.loads(response.content[0].text)
```

> **Alternative:** Use DeepSeek-V3 as the judge instead of Claude Haiku — it costs ~$0.11 vs ~$0.24 for 750 calls, and since it generated the golden answers it has perfect familiarity with the expected schema.

### Step 4 — Statistical Significance

#### What this step does (in plain English)

If your fine-tuned Qwen model gets 97.3% policy accuracy and the base model gets 74.1%, that looks like a big improvement — but is it real, or just luck from this particular 150-row test set? What if you picked a different 150 rows and the results flipped?

Statistical significance testing answers this question. We use **bootstrapping**: randomly resample the 150 results 1,000 times and recalculate the accuracy each time. This gives us a range (called a 95% confidence interval) — for example, "the fine-tuned model's accuracy is 97.3% ± 1.8%." If the confidence intervals of two models don't overlap, the difference is statistically real — not just noise.

This turns your results from "our model seems better" into "our model is provably better with 95% statistical confidence" — a critical distinction for any serious technical portfolio.

Bootstrap confidence intervals (n=1,000 resamples) on accuracy metrics to confirm fine-tuned vs. base improvements are statistically significant (p < 0.05).

### Step 5 — Reporting Pipeline (Before / After / Cross-Comparison)

#### What this step does (in plain English)

Raw results files are not readable by a human or a recruiter. This step produces three clean report artifacts that tell the full evaluation story. They are generated automatically by a script that reads `eval_results.json` and formats the numbers into readable markdown tables.

**Report 1 — Pre-Training Baseline (`results/00_baseline_report.md`):**
Generated after Week 2 Step 1. Shows how the *untrained* base models perform on all key metrics — this is the "before" photo. Expected to be bad. That's the point.

```
## Baseline Performance (Zero-Shot, No Fine-Tuning)

| Metric                  | Base Qwen-2.5-7B | Base Llama-3.2-3B |
|-------------------------|------------------|-------------------|
| JSON Validity Rate      | ~70%             | ~55%              |
| Intent Accuracy         | ~76%             | ~68%              |
| Gatekeeper Status Acc.  | ~74%             | ~65%              |
| Slot Extraction F1      | ~70%             | ~62%              |
| Hallucination Rate      | ~12%             | ~18%              |
| p50 Latency             | ~200ms           | ~100ms            |
```

**Report 2 — Post-Training Summary (`results/eval_summary.json`):**
Generated in Week 3. Contains all 5 model variants × all metrics with bootstrap 95% confidence intervals.

**Report 3 — Cross-Comparison Delta Table (`results/cross_comparison_table.md`):**
The money table — shows the improvement from fine-tuning and the gap to teacher. This is the "before/after" story for your portfolio README.

```
## Before vs. After Fine-Tuning

| Metric               | Base Qwen | FT Qwen (Track A) | Δ Improvement | Teacher Gap |
|----------------------|-----------|-------------------|---------------|-------------|
| JSON Validity        | 70%       | 100%              | +30pp         | 0pp         |
| Intent Accuracy      | 76%       | ≥97%              | +21pp         | <2pp        |
| Gatekeeper Accuracy  | 74%       | ≥97%              | +23pp         | <2pp        |
| Hallucination Rate   | 12%       | <1%               | −11pp         | <0.5pp      |

| Metric               | Base Llama | FT Llama (Track B) | Δ Improvement | Teacher Gap |
|----------------------|------------|---------------------|---------------|-------------|
| JSON Validity        | 55%        | ≥98.5%              | +43.5pp       | <1.5pp      |
| Intent Accuracy      | 68%        | ≥91.5%              | +23.5pp       | <8pp        |
| TTFT                 | ~80ms      | <150ms              | ✓ target met  | —           |
```

### Deliverables

- `results/00_baseline_report.md` — pre-training performance of base models
- `results/eval_results.json` — full 5-model × 150-row evaluation with judge scores
- `results/eval_summary.json` — aggregated metrics with bootstrap 95% CIs
- `results/cross_comparison_table.md` — before/after delta table for portfolio README
- Vercel Postgres populated with all row-level results
- LangSmith trace dashboard link (judge call logs)

#### What you now have after Week 3

You now have a complete, rigorous evaluation of all 5 model variants — with automated metrics, LLM-as-a-Judge scores, latency benchmarks, and statistically validated results. You can prove exactly how much better the fine-tuned models are vs. the base models, and how they compare to the expensive teacher. This is the core evidence base for your portfolio story.

---

## Week 4: Dashboard + Portfolio Packaging

**Objective:** Build the Next.js analytics dashboard and publish the open-source repository.

**Quality Gate 4:** Data tables must clearly show cost-performance tradeoffs and the operational benefits of the multi-tiered fleet to technical reviewers and hiring managers.

### Dashboard Architecture (Next.js + Vercel Postgres)

#### What this step does (in plain English)

Raw JSON files and terminal outputs aren't impressive to a hiring manager — a live, interactive website is. This is where we build the portfolio-facing front end: a Next.js web app deployed publicly on Vercel that visualizes all the evaluation results.

**Next.js** is a React-based web framework (JavaScript/TypeScript) — it's what most modern web apps are built with. **Vercel** is a hosting platform that makes deploying Next.js apps incredibly simple — you push to GitHub and it auto-deploys. Since we're already storing evaluation results in **Vercel Postgres** (a cloud database), the dashboard can query those results directly.

You don't need to be a front-end developer to build this. The dashboard has 5 pages, each serving a specific purpose for the portfolio story.

```
dashboard/
├── app/
│   ├── page.tsx                 # Home: KPI summary + fleet overview
│   ├── rows/[id]/page.tsx       # Row inspector: side-by-side outputs
│   ├── cost/page.tsx            # Cost calculator: monthly volume → cost table
│   ├── multilingual/page.tsx    # Accuracy by language (EN/ES/FR/AR)
│   └── api/
│       ├── eval/route.ts        # Fetch eval results from Vercel Postgres
│       └── cost/route.ts        # ROI calculation endpoint
├── components/
│   ├── ModelComparisonChart.tsx # Plotly/Recharts bar charts
│   ├── RowInspector.tsx         # Side-by-side JSON output viewer
│   ├── CostCalculator.tsx       # Slider → monthly cost breakdown
│   └── TriageRouterDemo.tsx     # Live demo: type a dispute → see routing decision
└── lib/
    └── db.ts                    # Vercel Postgres client
```

**Dashboard Pages:**
- **Page 1 — KPI Summary:** Bar charts comparing all 5 models on JSON validity, policy accuracy, latency, hallucination rate
- **Page 2 — Row Inspector:** Dropdown selects any test row; displays all 5 model outputs side-by-side with ground truth
- **Page 3 — Cost Calculator:** Slider for monthly dispute volume → table showing DeepSeek-V3 vs. Qwen-7B vs. Llama-3B monthly cost + ΔROI
- **Page 4 — Multilingual Breakdown:** Accuracy by language for Track A
- **Page 5 — Triage Router Demo:** Type a customer message → see router decision (Track A or Track B) + Track A/B model response

### Repository Structure

```
ecommerce-return-gatekeeper/
├── README.md
├── prd.md
├── plan.md
├── data/
│   ├── bitext_seeds.jsonl
│   └── distilled_dataset.jsonl
├── scripts/
│   ├── 01_generate_dataset.py      # Bitext filter + DeepSeek-V3 generation
│   ├── 02_validate_dataset.py      # Schema validation gate
│   ├── 03_evaluate_models.py       # Multi-model benchmark runner
│   └── 04_judge_outputs.py         # LLM-as-a-Judge scoring
├── configs/
│   ├── qwen_qlora.yaml
│   └── llama_qlora.yaml
├── router/
│   └── triage_router.py            # Dynamic Triage Router
├── results/
│   ├── baseline_results_qwen.json
│   ├── baseline_results_llama.json
│   └── eval_results.json
└── dashboard/                      # Next.js app
    ├── app/
    ├── components/
    └── lib/
```

### README Must-Haves

#### What this step does (in plain English)

The README is the first thing a recruiter or engineering manager sees when they open your GitHub repository. It needs to tell the entire story of the project — what problem it solves, how it works, what the results were — in 3 minutes of reading.

Think of the README as the executive summary of your project. It should be skimmable (lots of tables and diagrams), evidence-heavy (actual numbers, not vague claims), and end with a clear call to action (link to the live dashboard).

- Architecture diagram (triage router + dual fleet)
- Cost comparison table (frontier vs. self-hosted at scale)
- Before/after KPI table (base vs. fine-tuned)
- ΔROI calculation with example monthly volumes
- "Lessons Learned / Design Trade-offs" section
- Link to live dashboard (Vercel)
- Links to model cards on HF Hub

### Deployment

#### What this step does (in plain English)

The last step is making everything publicly accessible. There are two things to publish:

1. **The fine-tuned model weights to HuggingFace Hub.** This is how you make your trained models publicly available for others to download and use. Each model gets its own "model card" — a page on HuggingFace that describes what the model does, how it was trained, and what the results were. Having your own models on HuggingFace is a strong portfolio signal.

2. **The Next.js dashboard to Vercel.** One command deploys the entire web app to a public URL. This becomes the live demo link you put in your resume, LinkedIn, and portfolio.

```bash
# Publish models to Hugging Face Hub
huggingface-cli upload <your-username>/qwen-2.5-7b-ecommerce-gk ./output/qwen-2.5-7b-ecommerce-gk
huggingface-cli upload <your-username>/llama-3.2-3b-ecommerce-gk ./output/llama-3.2-3b-ecommerce-gk

# Deploy Next.js dashboard to Vercel (connects to Vercel Postgres automatically)
vercel --prod ./dashboard
```

---

## Deliverables Checklist

### Week 1
- [ ] `data/bitext_seeds.jsonl` — 2,000 filtered seed rows
- [ ] `data/distilled_dataset.jsonl` — ≥1,500 validated golden rows
- [ ] `data/train.jsonl` (1,200 rows), `data/val.jsonl` (150 rows), `data/test.jsonl` (150 rows, held out)
- [ ] **Quality Gate 1 passed:** ≥1,500 validated rows confirmed

### Week 2
- [ ] `results/baseline_results_qwen.json` — zero-shot Qwen baseline (150 rows)
- [ ] `results/baseline_results_llama.json` — zero-shot Llama baseline (150 rows)
- [ ] `output/qwen-2.5-7b-ecommerce-gk/` — fine-tuned Qwen adapter weights
- [ ] `output/llama-3.2-3b-ecommerce-gk/` — fine-tuned Llama adapter weights
- [ ] `router/triage_router.py` — triage router prototype
- [ ] W&B training run links for both models
- [ ] MMLU guard metrics (pre/post for both models, <5pt drop threshold)
- [ ] **Quality Gate 2 passed:** Loss curves stable, no overfitting detected

### Week 3
- [ ] Vercel Postgres populated with all eval results
- [ ] `results/eval_results.json` — full 5-endpoint × 150-row evaluation
- [ ] `results/eval_summary.json` — aggregated metrics with bootstrap 95% CIs
- [ ] LLM-as-a-Judge scores per row
- [ ] **Quality Gate 3 passed:** Track A = 100% JSON + ≥97% gatekeeper_status; Track B TTFT <150ms

### Week 4
- [ ] Live Next.js dashboard deployed on Vercel (URL in README)
- [ ] Triage Router Demo page functional
- [ ] Public GitHub repository (all scripts, configs, results open-sourced)
- [ ] Qwen model card on Hugging Face Hub
- [ ] Llama model card on Hugging Face Hub
- [ ] README complete with results narrative, ΔROI table, and design trade-offs
- [ ] **Quality Gate 4 passed:** Dashboard clearly shows cost-performance tradeoffs

#### What you now have after Week 4 — the complete project

You have a fully working, publicly deployed AI system with:
- A curated synthetic training dataset (built with a frontier AI)
- Two fine-tuned open-source models (published on HuggingFace)
- A rigorous multi-model evaluation (with statistical significance)
- A live web dashboard showing cost-performance tradeoffs
- A clean GitHub repository with everything documented

This is not a tutorial project — this is an original system you designed, built, and validated from scratch. The story it tells: you can take a business problem (cost), design an AI architecture to solve it, build the data pipeline, train and evaluate the models, and ship a working product — end to end.
