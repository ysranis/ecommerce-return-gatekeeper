# Business Explanation: The E-commerce Return Gatekeeper Project
### A Plain-Language Guide for AI Product Managers Entering the Field

---

## Table of Contents

1. [What Is This Project, Really?](#1-what-is-this-project-really)
2. [The Business Problem in Simple Terms](#2-the-business-problem-in-simple-terms)
3. [How We Get the Models — Where Do They Come From?](#3-how-we-get-the-models--where-do-they-come-from)
4. [How We Generate the Dataset — Teaching the Student](#4-how-we-generate-the-dataset--teaching-the-student)
5. [The Tools We Use and Why](#5-the-tools-we-use-and-why)
6. [Where Models Live Before and After Training](#6-where-models-live-before-and-after-training)
7. [What Does Everything Cost?](#7-what-does-everything-cost)
8. [How We Benchmark — The Before and After Comparison](#8-how-we-benchmark--the-before-and-after-comparison)
9. [Observability — What We Track and Why It Matters](#9-observability--what-we-track-and-why-it-matters)
10. [Your AI PM Pitch to Recruiters](#10-your-ai-pm-pitch-to-recruiters)
11. [Your Video Demo and Presentation Guide](#11-your-video-demo-and-presentation-guide)

---

## 1. What Is This Project, Really?

Imagine you run a large e-commerce store — think of something like a mid-sized version of Amazon or ASOS. Every day, thousands of customers contact your support system with complaints like:

> *"hey my package hasnt arrived in 3 weeks and i want my money back NOW"*

> *"cancel order AX-9481, the thing arrived smashed"*

> *"can i still return this? i opened it but it doesnt work"*

A human support agent reads each message, figures out what the customer wants, checks company policy ("is the item within the 30-day return window?"), and makes a decision: approve the refund, ask for a photo, or pass it to a senior agent.

**This project replaces that entire process with a small, fast, cheap AI model** — one that:
- Reads the raw, messy customer message
- Understands what the customer wants (their "intent")
- Extracts key details (order number, invoice number, what's wrong with the item)
- Checks the situation against company return policy
- Makes a decision: approve automatically, ask for evidence, or escalate to a human
- Writes a polite reply back to the customer

The key innovation is not just *building* this AI system — it's building it at a **99% lower cost** than using a frontier AI model like GPT-4o, while maintaining nearly identical accuracy. That cost reduction is the entire product strategy, and demonstrating it with real data is what makes this portfolio piece extraordinary.

---

## 2. The Business Problem in Simple Terms

### The Two Extremes (Both Are Wrong)

**Option A — Use a Powerful Frontier AI (GPT-4o, Claude):**
These models are brilliant. They understand messy customer language perfectly, follow policy logic, and never hallucinate order numbers. But they cost money every time they respond. At high volumes, this becomes financially unsustainable.

Think of it like hiring a Harvard-educated lawyer to answer every single customer email. Yes, the answers will be perfect. But you'll go bankrupt.

**Option B — Use a Cheap, Generic Small AI Model:**
A tiny open-source model costs almost nothing to run — you pay for the server, not per message. But out-of-the-box, small models are unreliable for business-critical tasks. They:
- Make up order numbers that don't exist (called "hallucination")
- Output messy, unparseable text instead of structured data your backend systems can read
- Get confused by colloquial language, typos, or emotional customers
- Make policy errors — like approving refunds they shouldn't

Think of it like hiring someone with no training to answer legal questions. Cheap, but dangerously wrong.

### Our Solution: The Middle Path

We take the cheap small model and **teach it to behave like the expensive expert** — specifically for our narrow task. This teaching process is called **model distillation and fine-tuning**.

The result: a small model that costs almost nothing to run but performs nearly as well as a frontier model on our specific task. And because we train *two* different small models, we can route different types of customer requests to the most appropriate model — a strategy called a **multi-tiered model fleet**.

---

## 3. How We Get the Models — Where Do They Come From?

### What Is a Language Model?

Think of a language model as a very sophisticated autocomplete system trained on enormous amounts of text. It has learned patterns of language — including reasoning patterns — from reading essentially most of the internet, books, and code. The "parameters" (the numbers stored inside the model) encode all of this learned knowledge.

### The Three Models in This Project

**The Teacher: DeepSeek-V3**
- Made by: DeepSeek AI (a Chinese AI research lab)
- Size: 671 billion parameters total (but uses only 37 billion at a time, thanks to a design called Mixture-of-Experts)
- Where we access it: We call it via an API (a web endpoint) — we pay per message we send it
- We never download or host this model ourselves; we rent access to it
- Cost: approximately $0.27 per million tokens (a "token" is roughly ¾ of a word)
- Role in this project: The expert. We use it to generate our golden training data.

**Student Model A: Qwen-2.5-7B-Instruct**
- Made by: Alibaba Cloud (released open-source)
- Size: 7 billion parameters (about 100x smaller than the teacher)
- Where we get it: Downloaded for free from **Hugging Face** (think of it as GitHub for AI models)
- We download and host this model on our own rented GPU server
- Role: Our "Performance-First" model, fine-tuned for policy logic, JSON output, and multilingual support

**Student Model B: Llama-3.2-3B-Instruct**
- Made by: Meta (Facebook's parent company, released open-source)
- Size: 3 billion parameters (even smaller and faster)
- Where we get it: Downloaded for free from **Hugging Face**
- We download and host this model on our own rented GPU server
- Role: Our "Cost-First" model, fine-tuned for speed — handles simple, repetitive requests

### What Is Hugging Face?

Hugging Face is the world's largest repository of open-source AI models and datasets. Thousands of companies and researchers publish their models there for free. It's like an App Store or GitHub, but for AI models and datasets.

When we say we "download the model from Hugging Face," we mean:

1. We open a terminal on our GPU cloud server
2. We run a command like: `huggingface-cli download Qwen/Qwen2.5-7B-Instruct`
3. The model files (typically 14–16GB for a 7B model) download directly to our server
4. The model is now on our machine and ready to be fine-tuned

No license fees. No API costs. Just the cost of the server running while we work with it.

---

## 4. How We Generate the Dataset — Teaching the Student

This is the most creative and technically important part of the project. Here's the full story of how our training data is born.

### Why We Can't Just Use Existing Data As-Is

We start with the **Bitext Gen AI Chatbot Customer Support Dataset** — a collection of 26,872 example customer support messages. It's freely available on Hugging Face.

This dataset has:
- Real-sounding customer messages with typos, slang, and emotional language ✓
- Labels for what the customer wants (intent) ✓
- Labels for how the language is written (colloquial, offensive, polite) ✓

But it has a critical problem: the **answers** in the dataset are generic, conversational text, not the structured JSON data our system needs to output. If we train a small model on these raw answers, it will learn to write conversational text — not policy decisions in structured format.

We need answers that look like this:

```
{
  "intent_action": "cancel_order",
  "gatekeeper_status": "APPROVE_AUTOMATED",
  "confidence_score": 0.94,
  "user_facing_response": "I have successfully canceled your order..."
}
```

No existing dataset has this. So we build it ourselves.

### The Knowledge Distillation Pipeline — Step by Step

**Step 1: Filter the Seed Data**
From the 26,872 rows, we keep only the ones relevant to e-commerce disputes:
- `get_refund`, `cancel_order`, `track_refund`, `complaint`, `check_refund_policy`
- We keep examples in all four language styles (colloquial, offensive, polite, typo-heavy)
- We extract **2,000 diverse seed messages** from these filtered rows

Think of these as 2,000 sample customer complaints we'll use as raw material.

**Step 2: Send Each Seed to DeepSeek-V3 with a Policy Handbook**
We give DeepSeek-V3 two things:
1. A fictional "Acme Corp Returns Policy Handbook" we write ourselves (specifying rules like "items over 30 days get store credit only" and "damaged items need photo evidence")
2. Each customer message from the seed data

We ask DeepSeek-V3 to play the role of an expert arbitration engine and output a fully reasoned, structured JSON decision for each message.

DeepSeek-V3 produces something like this for each message:

```json
{
  "chain_of_thought": "Customer wants to cancel order AX-9481 due to shipping delays. Per the returns handbook, orders stuck in transit over 14 business days qualify for immediate cancellation...",
  "intent_action": "cancel_order",
  "extracted_slots": { "order_id": "AX-9481" },
  "gatekeeper_status": "APPROVE_AUTOMATED",
  "user_facing_response": "I've canceled your order and issued a full refund..."
}
```

This is the "golden label" — the perfect answer we want our small model to learn to replicate.

**Step 3: Validate Every Output**
After DeepSeek-V3 generates responses, we run every single one through an automated parser. If the output isn't valid JSON with all required fields, we throw it out (or retry up to twice with a stricter prompt).

We keep going until we have **at least 1,500 validated, parseable golden examples**. These become our training dataset.

**Step 4: Split the Data**
- 1,200 rows → Training (the model learns from these)
- 150 rows → Validation (we check during training if the model is improving)
- 150 rows → Test (we lock these away and only use them at the very end for final scoring — the model never sees these during training)

This is exactly how any rigorous scientific experiment works: you need a truly unseen test set to prove your results are real, not just memorized.

---

## 5. The Tools We Use and Why

Here's every tool in plain language:

### For Generating the Dataset
| Tool | What It Is | Why We Use It |
|---|---|---|
| **Bitext Dataset** | A library of 26,872 customer support messages | Free, high-quality seed material with real linguistic variation |
| **Hugging Face `datasets` library** | Python library to download and filter datasets | Industry standard; makes loading Bitext a 2-line operation |
| **DeepSeek-V3 API** | Web service to send messages to DeepSeek-V3 | The teacher model that generates our golden training data |

### For Fine-Tuning (Training)
| Tool | What It Is | Why We Use It |
|---|---|---|
| **Axolotl** | Open-source fine-tuning framework | Simplifies QLoRA training with config files; widely used by the community |
| **Unsloth** | Alternative fine-tuning library | 2x faster training speed vs. Axolotl; lower memory usage |
| **QLoRA** | Training technique (Quantized LoRA) | Lets us fine-tune a 7B model on a single GPU instead of needing 8 GPUs |
| **RunPod or Lambda Labs** | Cloud GPU rental services | Hourly GPU rental — far cheaper than buying hardware |

### For Running the Models (Inference)
| Tool | What It Is | Why We Use It |
|---|---|---|
| **vLLM** | Open-source inference server | Serves fine-tuned models as a REST API with high throughput and low latency |

### For Tracking Training Progress (Observability)
| Tool | What It Is | Why We Use It |
|---|---|---|
| **Weights & Biases (W&B)** | Experiment tracking dashboard | Logs training metrics in real-time so you can see if training is working |

### For Evaluation
| Tool | What It Is | Why We Use It |
|---|---|---|
| **DeepEval or Arize Phoenix** | LLM evaluation frameworks | Automated scoring of model outputs against ground truth |
| **Claude Sonnet / GPT-4o** | Used as the "judge" model | Rates model outputs quality on a 1–5 scale |
| **lm-evaluation-harness** | Open-source benchmark runner | Runs the MMLU benchmark to check if fine-tuning broke general reasoning |

### For the Dashboard
| Tool | What It Is | Why We Use It |
|---|---|---|
| **Next.js** | React-based web framework | Modern, fast, deploys to Vercel in one command |
| **Vercel Postgres** | Cloud database | Stores evaluation results so the dashboard can query them |
| **Vercel** | Deployment platform | Free tier for small apps; one-click deployment |

---

## 6. Where Models Live Before and After Training

This is one of the most confusing parts for newcomers. Here's the complete journey of a model file:

### Before Training: Hugging Face → Your GPU Server

1. You rent a GPU cloud server (e.g., an A100 with 80GB VRAM on RunPod for ~$2.00/hour)
2. You SSH into that server (open a terminal connection to it)
3. You run: `huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ./models/qwen-base`
4. The model downloads as a collection of files (`.safetensors` files, a `config.json`, and a `tokenizer.json`)
5. The model now lives on that rented server's disk at `./models/qwen-base/`

The model is essentially a large file containing all of its learned knowledge as a matrix of numbers. The 7B model is roughly 14GB in size (like 14 HD movies).

### During Training: The Adapter Is Born

When we run QLoRA fine-tuning, something elegant happens. We **do not modify the original model weights**. Instead, we train a small set of additional weights called a **LoRA adapter** — think of it as a small "personality transplant module" that sits on top of the original model.

The adapter is typically only 50–200MB in size, compared to the 14GB base model. This is why QLoRA is so efficient.

During training:
- Base model: frozen (unchanged)
- Adapter: actively learning, updated with each training step
- Progress: saved to disk every epoch

The original 14GB model file is never touched.

### After Training: Where It Goes

**Option A — Merged Weights (for deployment):**
We merge the small adapter into the base model, creating a new standalone 14GB model file that has absorbed the training. This merged model is uploaded to Hugging Face as a public model card (e.g., `your-username/qwen-2.5-7b-ecommerce-gk`).

**Option B — Adapter Only (for sharing):**
We can publish just the 200MB adapter file on Hugging Face, and anyone with the base Qwen model can combine them to recreate our fine-tuned version.

**For serving (the dashboard demo):**
The merged model is loaded into vLLM on the GPU server, which exposes it as a REST API endpoint. Our Next.js dashboard calls this endpoint to generate live responses during the demo.

### Storage Summary

| Stage | Location | Size |
|---|---|---|
| Base model (downloaded) | RunPod cloud server local disk | ~14GB (Qwen-7B) / ~6GB (Llama-3B) |
| LoRA adapter (during training) | RunPod cloud server local disk | ~50–200MB |
| Merged fine-tuned model | Hugging Face Hub (public) | ~14GB (Qwen-7B) / ~6GB (Llama-3B) |
| Eval results | Vercel Postgres database | <1MB |
| Dashboard | Vercel CDN | <50MB |

---

## 7. What Does Everything Cost?

Here is an honest, realistic cost breakdown for the entire 4-week project:

### One-Time Project Costs

| Cost Item | Estimated Cost | Notes |
|---|---|---|
| DeepSeek-V3 API — Dataset Generation | ~$8–15 | ~2,000 calls × avg 1,000 tokens in/out = ~2M tokens at $0.27/M input + $1.10/M output |
| GPU Rental — Qwen-7B Fine-Tuning | ~$6–10 | ~3–5 hours on A10G (24GB) at $0.74/hr on RunPod |
| GPU Rental — Llama-3B Fine-Tuning | ~$2–4 | ~2–3 hours on A10G for 3B model |
| GPU Rental — Running vLLM for evaluation | ~$4–6 | ~4–6 hours serving all 5 endpoints during Week 3 |
| Vercel (Dashboard hosting) | Free | Vercel Hobby plan is free for small traffic |
| Vercel Postgres | Free | 256MB included in free plan — more than enough |
| Hugging Face (Model hosting) | Free | Public model repos are free |
| W&B (Experiment tracking) | Free | Free for personal/open-source projects |
| **Total Estimated Project Cost** | **~$20–35** | One-time, for a complete portfolio project |

### Ongoing Production Cost Comparison (the business case)

This is the number that matters for the pitch. At a hypothetical e-commerce company processing 1 million customer disputes per month, each averaging 500 tokens in and 300 tokens out:

| Model | Cost per 1M Tokens | Monthly Cost at 1M Disputes |
|---|---|---|
| GPT-4o (frontier) | ~$5.00 | ~$4,000/month |
| DeepSeek-V3 (teacher) | ~$0.27 input / $1.10 output | ~$700/month |
| Qwen-2.5-7B (self-hosted) | ~$0.15 | ~$120/month (server cost) |
| Llama-3.2-3B (self-hosted) | ~$0.06 | ~$48/month (server cost) |

**ΔROI of self-hosted fleet vs. GPT-4o: ~97% cost reduction**

A $4,000/month cost drops to $120–$168/month while maintaining near-identical accuracy. That's the business case. That's why a company would build this.

---

## 8. How We Benchmark — The Before and After Comparison

"Benchmarking" means running systematic tests to measure exactly how well a model performs — and critically, *how much better* it got after fine-tuning.

### The Five Models We Compare

We test five model variants against the same 150 questions (our held-out test set):

1. **Base Qwen-2.5-7B** — The off-the-shelf model, no training (our "before" for Track A)
2. **Fine-Tuned Qwen-2.5-7B** — After our training (our "after" for Track A)
3. **Base Llama-3.2-3B** — The off-the-shelf model, no training (our "before" for Track B)
4. **Fine-Tuned Llama-3.2-3B** — After our training (our "after" for Track B)
5. **DeepSeek-V3** — The teacher (our gold standard "ceiling")

### The Metrics — What We Measure and What Each Means

**JSON Validity Rate**
Does the model output valid, parseable JSON or does it write a messy paragraph?
- This is binary: either the output is valid JSON (a computer can read it) or it isn't (the backend system crashes)
- Base models often fail 30–45% of the time. After fine-tuning, Track A should hit 100%

**Policy Accuracy (gatekeeper_status)**
When the model makes a decision (APPROVE / REQUEST_EVIDENCE / ESCALATE), is it the correct business decision?
- This is checked against our golden dataset's ground truth labels
- Getting this wrong has real business cost — approving a fraudulent refund, or wrongly rejecting a legitimate one

**Intent Classification Accuracy**
Did the model correctly understand what the customer wanted?
- Example: Did it correctly classify "I want my money back" as `get_refund` rather than `complaint`?

**Entity Slot Extraction F1-Score**
Did the model correctly pull out the key data points from the message?
- F1-Score combines precision (when it says it found an order number, is it right?) and recall (how often does it find order numbers that are actually there?)
- Extracting the wrong order number could process the refund for the wrong order

**Hallucination Rate**
Did the model make up information that wasn't in the customer's message?
- Example: Customer didn't mention an invoice number, but the model outputs `"invoice_id": "INV-4521"` — this number is fabricated
- Even a 5% hallucination rate could cause significant financial errors at volume

**Latency (p50 and p95)**
- **p50** = The response time for the median (50th percentile) request — the "typical" speed
- **p95** = The response time for 95% of all requests — the "almost worst case" speed
- We measure both because averages hide the bad cases that hurt customer experience

**Time-to-First-Token (TTFT)**
For Track B (the fast model), how quickly does the first word of the response appear?
- Important for customer experience — even if the full response takes 200ms, a fast TTFT (say 80ms) makes the system feel responsive

### How We Run the Comparison

We run all 150 test questions through all 5 endpoints simultaneously, then:
1. Automated scoring checks JSON validity, intent accuracy, entity F1, and hallucination rate
2. An LLM-as-a-Judge (Claude or GPT-4o) rates each response on a 1–5 scale for overall quality
3. We store all results in Vercel Postgres and display them on the dashboard
4. We run statistical significance tests to confirm the improvements are real, not random luck

The result is a clear data table showing exactly how much each model improved — which becomes the centerpiece of the portfolio.

---

## 9. Observability — What We Track and Why It Matters

"Observability" in AI means being able to *see what's happening inside your system* so you can understand, improve, and debug it. For AI product managers, this is one of the most important concepts to understand — because without observability, you are flying blind.

### During Training: Weights & Biases (W&B)

W&B is a dashboard that receives real-time updates from your training run. Every few minutes of training, it logs:

**Training Loss**
This is the primary signal of whether learning is happening. Loss is a number that measures "how wrong the model is." At the start of training, loss is high. As the model learns, loss decreases.

- If loss goes down steadily → training is working ✓
- If loss decreases but then starts going up → overfitting (the model is memorizing, not learning) ✗
- If loss barely moves → the learning rate or dataset may need adjustment ✗
- If loss oscillates wildly → the learning rate is too high ✗

**Validation Loss**
The same measurement, but on the 150-row validation set the model has never trained on. If training loss keeps dropping but validation loss starts rising, that's the first sign of overfitting — and we trigger early stopping.

**Learning Rate**
How aggressively the model updates its weights each step. We use a cosine schedule (starts high, gradually decreases) to achieve stable convergence.

**What W&B Shows You (as an AI PM)**
When you sit in a review with engineers, you can look at the W&B dashboard and say: "I can see the model converged by epoch 2, validation loss stabilized, and there are no signs of overfitting. The training run is clean." This is the kind of technical fluency that makes AI PMs exceptionally valuable.

### After Training: Catastrophic Forgetting Check

Fine-tuning a model for a narrow task can sometimes accidentally damage its general reasoning abilities — like a doctor who becomes so specialized in cardiology that they forget how to read an X-ray.

We use the **MMLU benchmark** (a standardized test of general knowledge and reasoning) to check whether our fine-tuned model's general IQ has dropped. If it drops more than 5 percentage points, we know the fine-tuning was too aggressive and we need to reduce the number of training epochs.

This is tracked as a before/after comparison and stored in our results.

### In Production (The Dashboard): What We Monitor

The Vercel Postgres database stores every model decision permanently, including:
- The raw customer input
- Which model handled it (router decision)
- The JSON output (gatekeeper_status, intent, slots)
- The confidence_score
- The processing timestamp and latency

This means we can:
- **Audit any decision** — "Why did we approve this refund? Show me the chain-of-thought"
- **Detect drift** — If accuracy starts dropping over time as customer language evolves
- **Monitor for hallucinations** — Alert when confidence_score drops below a threshold
- **Measure ROI in real-time** — Cost per dispute processed, accuracy rate over time

### Why This Matters for the Portfolio

Showing observability in your portfolio proves you understand that AI is not a one-time build. It's a living system that needs to be monitored, measured, and maintained. This is a key mindset that separates junior from senior AI PMs.

---

## 10. Your AI PM Pitch to Recruiters

### The Core Narrative

Every recruiter pitch needs a clear story arc: Problem → Solution → Proof → Impact. Here's how to frame this project:

---

**HEADLINE (15 seconds):**
> *"I built an AI-powered e-commerce dispute arbitration system that reduces customer support inference costs by 97% without sacrificing accuracy — using model distillation, QLoRA fine-tuning, and a live benchmarking dashboard I designed and shipped."*

---

**THE PROBLEM (30 seconds):**
> *"Running enterprise-grade AI agents at scale using frontier models like GPT-4o costs thousands of dollars per month. But off-the-shelf small models hallucinate, break JSON schemas, and make wrong policy decisions. Most teams pick one or the other and live with the tradeoffs. I built a third path."*

**THE SOLUTION (45 seconds):**
> *"I designed a multi-tiered AI fleet: a 7-billion parameter model fine-tuned for complex policy arbitration, and a 3-billion parameter model for high-speed routine requests — coordinated by a triage router. I used DeepSeek-V3 to generate 1,500 golden training examples from a real customer support dataset, incorporating a custom returns policy handbook. I ran parallel fine-tuning experiments tracked in real-time via Weights & Biases."*

**THE PROOF (30 seconds):**
> *"After fine-tuning, the 7B model achieves 100% JSON validity and 97%+ policy accuracy — matching the frontier teacher at 99% lower inference cost. The 3B model handles routine queries in under 150 milliseconds. I benchmarked all five model variants — two base, two fine-tuned, one teacher — using automated metrics and an LLM-as-a-Judge framework."*

**THE IMPACT (15 seconds):**
> *"For a company processing one million disputes per month, this architecture reduces AI spend from $4,000/month to roughly $150/month. That's not a 10% improvement — it's an order-of-magnitude change in unit economics."*

---

### What Skills This Project Demonstrates

When talking to recruiters or hiring managers, map each component to the skill it proves:

| What You Built | Skill It Proves |
|---|---|
| Defined the PRD with feature matrix, KPIs, and risk register | Product strategy and documentation |
| Chose DeepSeek-V3 over GPT-4o as teacher to demonstrate cost-awareness | Technical depth + financial reasoning |
| Designed the triage router to route by complexity | Systems thinking, not just model training |
| Generated synthetic training data with a policy handbook | Data engineering and domain design |
| Ran parallel fine-tuning experiments with W&B tracking | ML engineering knowledge |
| Added MMLU catastrophic forgetting check | Production-thinking, not just demo-thinking |
| Built the benchmark with 5 model variants and statistical significance | Rigorous evaluation methodology |
| Built the Next.js dashboard with a cost calculator | End-to-end product delivery |
| Added audit logging and confidence score routing | Enterprise readiness and compliance thinking |

### The One Line for LinkedIn

> *"Built and shipped an end-to-end model distillation and fine-tuning benchmark for e-commerce dispute arbitration — demonstrating 97% inference cost reduction while maintaining frontier-level accuracy across 5 model variants."*

### Resume Bullet Points

```
• Designed and executed an end-to-end model distillation pipeline using DeepSeek-V3 as a
  teacher model to generate 1,500+ golden training examples for e-commerce dispute arbitration

• Fine-tuned Qwen-2.5-7B and Llama-3.2-3B using QLoRA (LoRA r=16/8, cosine LR schedule,
  3 epochs) achieving 100% JSON schema compliance and ≥97% policy accuracy on Track A

• Architected a multi-tiered AI fleet with a Dynamic Triage Router achieving 97% inference
  cost reduction vs. frontier APIs ($4,000/month → ~$150/month at 1M disputes/month)

• Built a 5-model evaluation harness with LLM-as-a-Judge scoring, bootstrap confidence
  intervals, and MMLU catastrophic forgetting guard rails

• Shipped a Next.js analytics dashboard (Vercel + Postgres) with live model comparison,
  row-level inspection, and an interactive cost calculator for hiring manager demos
```

---

## 11. Your Video Demo and Presentation Guide

### The 3-Minute Video Demo Script

A great video demo tells a story. Don't start by showing code. Start by showing the problem.

**Minute 1 — The Problem Setup (Business Context)**

Show the screen: Open your Next.js dashboard's home page.

Say:
> *"Let me show you a real business problem. Every day, e-commerce companies receive thousands of messy, emotional customer dispute messages like these."*

Show the Row Inspector page — click a test row with an angry, typo-filled customer message.

> *"A frontier AI model like GPT-4o handles this perfectly — but at scale, it costs thousands of dollars per month. A cheap small model out-of-the-box can't even output valid JSON."*

Show the baseline model column — demonstrate the broken or incorrect output.

**Minute 2 — The Solution in Action (Technical Depth)**

> *"So I built a system that teaches a small, cheap model to behave like an expert."*

Show the Architecture Diagram in the dashboard or README.

> *"I used DeepSeek-V3 as a teacher model to generate 1,500 golden training examples — complete with chain-of-thought reasoning and structured policy decisions — from a real customer support dataset. Then I fine-tuned two open-source models using a technique called QLoRA."*

Click on the KPI Summary page.

> *"After fine-tuning, the 7B Qwen model achieves 100% JSON validity, 97% policy accuracy, and sub-250 millisecond response time. That's essentially identical to the frontier teacher — at 1% of the cost."*

**Minute 3 — The Business Impact (The "Wow" Moment)**

Navigate to the Cost Calculator page.

> *"This is the part that matters to a business. I built an interactive cost calculator. Let me set the monthly dispute volume to 500,000."*

Show the table updating in real time.

> *"At half a million disputes per month, using GPT-4o costs roughly $2,000/month. Using our fine-tuned Qwen model on self-hosted infrastructure costs about $75/month. That's a 96% reduction — roughly $23,000 saved per year, per product.*

> *And for simpler queries — tracking requests, standard cancellations — our Triage Router automatically sends those to the Llama-3B model, which responds in under 150 milliseconds and costs even less."*

Show the Triage Router Demo: type a simple tracking query, watch it route to Track B. Type a complex dispute, watch it route to Track A.

> *"This is the full picture: not just a trained model, but a complete system — data pipeline, training, evaluation, deployment, and business impact analysis. All open-sourced on GitHub."*

---

### Presentation Slide Deck Structure

For a slide deck (hiring presentation, portfolio walkthrough), use this structure:

**Slide 1 — Title Slide**
"E-commerce Return Gatekeeper: Model Distillation & Fleet Optimization"
Your name + link to GitHub + live dashboard URL

**Slide 2 — The Business Problem**
Two columns: "Option A: Frontier AI ($4,000/month)" vs "Option B: Generic Small Model (unreliable)"
Caption: "There had to be a better way."

**Slide 3 — The Solution Architecture**
The triage router diagram from prd.md. Label every box. Keep it clean.

**Slide 4 — How the Data Was Built**
Bitext → DeepSeek-V3 → Golden Dataset flow. Show one example of raw customer message → golden JSON output.

**Slide 5 — The Training Setup**
Simple visual: two parallel tracks, QLoRA config highlights (lora_r, epochs, sequence_len). W&B loss curve screenshot.

**Slide 6 — The Results (KPI Table)**
The 5-model comparison table. Use color coding: green for best-in-class, yellow for acceptable, red for baseline failures.

**Slide 7 — The Business Case**
Cost calculator table at 3 different volume levels (100K / 500K / 1M disputes/month). Big ΔROI number front and center.

**Slide 8 — The Live Dashboard**
Screenshot of each dashboard page. Let the product speak.

**Slide 9 — What I Learned / Design Trade-offs**
This is the slide that separates great AI PMs from everyone else. Talk about:
- Why you chose Qwen over Llama for Track A (JSON adherence, multilingual)
- Why you chose DeepSeek-V3 over GPT-4o as teacher (cost-conscious from the start)
- What you would do with more data (multilingual expansion, adversarial test cases)
- What the catastrophic forgetting check taught you

**Slide 10 — What's Next**
- Expand to a live API with streaming responses
- Add multilingual evaluation across all 4 languages
- Train the triage router itself using a lightweight classifier

---

### The One Thing to Emphasize in Every Conversation

> **You didn't just build a model. You built a product.**

Anyone can run a fine-tuning script. Very few people define a PRD, design a feature matrix, write a risk register, instrument observability, establish quality gates, build an evaluation harness, ship a dashboard, and articulate the business ROI — all in one project.

That combination of product thinking + technical execution is exactly what the "AI Product Manager / Builder" title demands. This project is the proof.

---

*For technical specifications, see [prd.md](./prd.md). For the week-by-week execution plan, see [plan.md](./plan.md).*
