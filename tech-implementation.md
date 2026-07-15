# Technical Implementation Guide
**Project:** E-commerce Return Gatekeeper
**Last updated:** 2026-07-14
**Week 1 status:** ✅ Complete
**Week 2 status:** ✅ Complete — Training + baseline eval finished on RunPod RTX 4090
**Week 3 status:** ✅ Complete — 5-model benchmark + LLM-as-a-Judge scoring done
**Week 4 status:** ✅ Complete — Dashboard live at https://ecommerce-return-gatekeeper.vercel.app

This file is the single source of truth for every technical decision made in this project. It is updated after each step and each file is completed. Written so both non-technical readers and developers can follow along.

---

## How to Read This File

- **Plain English** explanations come first — no jargon assumed.
- **Technical details** follow in code blocks for developers.
- Each section is labeled with its week and step so you always know where you are in the project.

---

## Project Overview (Plain English)

We are building an AI system that automatically handles customer return and refund disputes for an e-commerce store. Instead of a human reading every complaint, the AI reads the message, decides what to do (approve the refund, ask for more evidence, or escalate to a human), and responds — in milliseconds.

The project is split into 4 weeks (all complete):
- **Week 1** — Build the training data (teach the AI what good answers look like) ✅ Complete
- **Week 2** — Train two AI models on that data ✅ Complete
- **Week 3** — Test and compare all models rigorously ✅ Complete
- **Week 4** — Build a web dashboard to show the results ✅ Complete

A **second training run** is planned to improve on the first-run gatekeeper accuracy scores and to compare results side-by-side.

---

## Repository Structure

```
ecommerce-return-gatekeeper/
├── scripts/           # Python scripts that do the actual work
│   └── lib/           # Shared helper code used by multiple scripts
├── data/              # All data files (seeds, training data, test data)
├── configs/           # Training configuration files
├── router/            # The logic that decides which AI model handles a request
├── results/           # Evaluation output and reports
├── dashboard/         # The Next.js web app (Week 4)
├── docs/              # Design specs and documentation
├── plan.md            # The full 4-week execution plan
├── prd.md             # Product requirements document
└── tech-implementation.md  # This file
```

**GitHub repo:** https://github.com/ysranis/ecommerce-return-gatekeeper (public)
**Live dashboard:** https://ecommerce-return-gatekeeper.vercel.app

**Branching rule:** Every new addition gets its own branch. We never work directly on `main`.

---

## Environment Setup

**What this means:** Before running any script, you need to set up a few things on your computer.

**`.env` file** — Create a file called `.env` in the project root. It holds your secret API keys and is never uploaded to GitHub.

```
DEEPSEEK_API_KEY=your_key_here
HF_TOKEN=your_huggingface_token_here
```

**Python virtual environment** — A `venv/` exists at the project root. Activate it or run via:
```bash
venv/bin/pytest -v          # run tests
venv/bin/python scripts/01_generate_dataset.py
```

**Python dependencies** — Install with:
```bash
pip install -r requirements.txt
```
Packages: `datasets`, `openai`, `python-dotenv`, `pytest`, `pytest-asyncio`

---

## Week 1 — Dataset Engineering ✅ Complete

**Status:** Code complete. PR #1 open at https://github.com/ysranis/ecommerce-return-gatekeeper/pull/1
**Next action:** Merge PR #1, then run the two scripts with real API keys to generate the data (~$2.75, ~60 min).

**Goal:** Create a high-quality set of training examples that teaches the AI what good dispute decisions look like.

**Why this matters:** AI models learn by example. Before we can train our models (Week 2), we need at least 1,500 perfectly-answered examples. This week is all about generating and validating those examples.

**Cost:** ~$2.75 in DeepSeek API fees. Runs entirely on your local Mac — no GPU or cloud instance needed.

**Tests:** 35/35 passing (`venv/bin/pytest -v`)

**How to run (once PR is merged and .env is set up):**
```bash
python scripts/01_generate_dataset.py   # ~30–60 min, ~$0.90
python scripts/02_label_dataset.py      # ~30–60 min, ~$1.85
```

---

### Step 1 & 2 — `scripts/01_generate_dataset.py`

**What this script does (plain English):**

1. Downloads 26,872 real customer support messages from a public AI dataset called Bitext
2. Filters down to only the 5 types of disputes we care about (refunds, cancellations, complaints, etc.)
3. Samples 400 messages per type = 2,000 seeds total
4. For each seed message, asks DeepSeek-V3 (a powerful AI) to rewrite it as a more realistic, messier, harder dispute — the kind real customers actually send
5. Saves each result immediately so if the script is interrupted, it can pick up where it left off

**Output:** `data/bitext_seeds.jsonl` — 2,000 rows

**Each row contains:**
- `seed_id` — a unique identifier for the row
- `original_message` — the original Bitext customer message
- `intent` — what the customer wants (e.g. `get_refund`)
- `synthetic_message` — the richer, more realistic version generated by DeepSeek-V3

**Key decisions:**
- Runs locally (no GPU needed — this is just API calls)
- 15 parallel API calls at a time (async) — fast but safe
- Checkpointing: saves each row as it's completed; if script crashes, restart picks up where it left off
- Discards Bitext's original answers entirely — we only want the customer questions, not their generic answers

**Target intents filtered:**
| Intent | Meaning |
|---|---|
| `get_refund` | Customer wants money back |
| `cancel_order` | Customer wants to cancel |
| `track_refund` | Customer asking where their refund is |
| `complaint` | General complaint |
| `check_refund_policy` | Customer asking about return rules |

---

### Step 3 & 4 — `scripts/02_label_dataset.py`

**What this script does (plain English):**

1. Reads the 2,000 synthetic messages from Step 1
2. For each message, asks DeepSeek-V3 to act as an expert arbitrator and produce a structured, detailed decision in a specific format (JSON)
3. Checks every answer: is it valid? Does it have all the required fields? If not, tries again (up to 2 retries)
4. If an answer is still bad after 2 retries, discards that row
5. Enforces a quality gate: we need at least 1,500 valid rows to proceed
6. Splits the validated rows into three sets: training (1,200), validation (150), and test (150)

**Output files:**
- `data/distilled_dataset.jsonl` — all validated golden rows
- `data/train.jsonl` — 1,200 rows used to train the models
- `data/val.jsonl` — 150 rows used to monitor training
- `data/test.jsonl` — 150 rows held out; **never used during training** (our final exam paper)

**Each golden row contains:**
- The customer's message
- The AI's step-by-step reasoning (`chain_of_thought`)
- What the customer wants (`intent_action`)
- Key details extracted from the message (order ID, invoice ID, etc.)
- A policy decision: approve / ask for evidence / escalate
- A confidence score
- The customer-facing response text

**Key decisions:**
- Retry logic: up to 2 retries with a stricter prompt before discarding a row
- Fixed random seed for splitting (same split every time — reproducible)
- Test set is **sacred** — never used for training or validation

---

### Shared Helper Files (`scripts/lib/`)

These are building blocks used by both scripts. Think of them as reusable tools in a toolbox.

#### `lib/deepseek_client.py`
**What it does:** Handles all communication with the DeepSeek AI API.
- Sends prompts, receives responses
- Runs up to 15 calls at the same time (async)
- If DeepSeek says "too many requests", waits and retries automatically (up to 3 times)

#### `lib/prompts.py`
**What it does:** Stores all the instructions (prompts) we give to DeepSeek-V3 in one place.
- The ACME Corp fictional returns policy handbook
- Instructions for generating synthetic messages (used in script 01)
- Instructions for producing structured JSON decisions (used in script 02)
- A stricter version of the labeling instructions used on retry attempts

**Why one place?** So that changing a prompt only requires editing one file, not hunting through multiple scripts.

#### `lib/bitext.py`
**What it does:** Filters and samples the Bitext dataset.
- `filter_bitext(rows, target_intents)` — keeps only rows matching the 5 target intents
- `sample_balanced(rows, target_intents, per_intent=400, seed=42)` — samples up to 400 rows per intent reproducibly

#### `lib/splitter.py`
**What it does:** Splits the validated dataset and enforces the quality gate.
- `split_dataset(rows, seed=42)` — shuffles and splits into train/val/test (1200/150/150)
- `check_quality_gate(rows, min_rows=1500)` — stops the pipeline if fewer than 1,500 valid rows exist

#### `lib/checkpoint.py`
**What it does:** Handles saving and loading progress mid-run.
- `load_checkpoint(path)` — reads what's already been done; returns empty list if nothing yet
- `append_row(path, row)` — safely saves one completed row to disk immediately

#### `lib/validator.py`
**What it does:** Checks whether a DeepSeek response is valid.
- Tries to parse the response as JSON
- Checks all required fields are present
- Returns a simple `True/False` result

**Required fields for a valid golden row (8 fields):**
`chain_of_thought`, `intent_action`, `extracted_slots`, `policy_evaluation`, `gatekeeper_status`, `confidence_score`, `fallback_escalation`, `user_facing_response`

---

### Error Handling (Week 1)

| What goes wrong | What the script does |
|---|---|
| DeepSeek rate limit (too many requests) | Waits and retries up to 3 times with increasing wait time |
| Bad JSON response from DeepSeek | Retries up to 2 times with a stricter prompt |
| Script crashes mid-run | Restart the same script — it picks up where it left off |
| Fewer than 1,500 valid rows | Script stops and tells you exactly how many were valid vs. discarded |
| Missing API key in `.env` | Script stops immediately with a clear error message |

---

### Design Spec

Full detailed spec: [`docs/superpowers/specs/2026-07-13-week1-dataset-engineering-design.md`](docs/superpowers/specs/2026-07-13-week1-dataset-engineering-design.md)

---

---

## Week 2 — Baseline Evaluation + Fine-Tuning ✅ Complete

**Status:** Complete. All 4 eval runs finished on RunPod RTX 4090. Results downloaded to `results/`.

**Goal:** Establish zero-shot baselines for both models, fine-tune them, then produce a clear "before vs. after" comparison report.

**Runs on:** RunPod A10G GPU instance (~$0.44/hr) — scripts 03 and 05/06 require a GPU. Script 04 runs locally on Mac.

---

### Step 1 — Zero-Shot Baseline + Fine-tuned Evaluation

#### `scripts/03_baseline_eval.py`

**What this script does (plain English):**

Before we train anything, we need to know how bad the untrained models are. We run all 150 held-out test rows through the raw, untouched Qwen-7B and Llama-3B models and ask them to produce the same structured JSON output we'll train them to produce. Most of the time they'll fail — wrong format, missing fields, hallucinated order IDs. That's expected. We record every failure. This is our "before" photo.

After fine-tuning (scripts 05 and 06), we run the same script again with the `--adapter-path` flag. This gives us the "after" photo with an identical output schema, enabling direct comparison.

**Metrics captured per model run:**

| Metric | What it measures |
|---|---|
| JSON validity rate | % of outputs that are valid, parseable JSON with all 8 required fields |
| Intent accuracy | % where model correctly classifies what the customer wants (e.g. `get_refund`) |
| Gatekeeper accuracy | % where model makes the correct policy decision (APPROVE / REQUEST_EVIDENCE / ESCALATE) |
| Slot F1 | How accurately `order_id` and `invoice_id` are extracted (micro-averaged F1) |
| Hallucination rate | % of rows where the model invents an order/invoice ID not present in the input |

**How to run:**

```bash
# On RunPod — BEFORE fine-tuning (baseline):
python scripts/03_baseline_eval.py --model Qwen/Qwen2.5-7B-Instruct --load-in-4bit
python scripts/03_baseline_eval.py --model meta-llama/Llama-3.2-3B-Instruct --load-in-4bit

# On RunPod — AFTER fine-tuning:
python scripts/03_baseline_eval.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --adapter-path output/qwen-2.5-7b-ecommerce-gk \
    --load-in-4bit

python scripts/03_baseline_eval.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --adapter-path output/llama-3.2-3b-ecommerce-gk \
    --load-in-4bit
```

**Output files (4 total):**
- `results/baseline_results_qwen.json` — Qwen base model, 150 rows + summary
- `results/baseline_results_llama.json` — Llama base model, 150 rows + summary
- `results/finetuned_results_qwen.json` — Qwen fine-tuned, 150 rows + summary
- `results/finetuned_results_llama.json` — Llama fine-tuned, 150 rows + summary

Each file contains full per-row data (ground truth, prediction, all metric flags) plus an aggregated `summary` block with the 5 metric rates.

**Key decisions:**
- `do_sample=False` (greedy decoding) — reproducible eval with no randomness
- Same `LABELING_SYSTEM_PROMPT` used in Week 1 generation — consistent zero-shot setup
- `--adapter-path` optional: omit for baseline, add for fine-tuned eval (same script, same output schema)
- 4-bit quantization via BitsAndBytes — fits 7B model on A10G 24GB VRAM

---

### Step 2 — Comparison Report

#### `scripts/04_compare_results.py`

**What this script does (plain English):**

Once all 4 eval runs are complete, this script loads the results and produces the central portfolio artifact: a side-by-side table showing exactly how much each metric improved after fine-tuning.

**Runs locally on Mac** — no GPU required, no ML dependencies (pure Python stdlib).

**How to run:**

```bash
# Locally, after copying results/ files from RunPod:
python scripts/04_compare_results.py
```

**Output files:**
- `results/comparison_report.json` — machine-readable delta data for the Week 4 dashboard
- `results/comparison_report.md` — human-readable Markdown table for the portfolio README

**Example output:**

| Metric | Baseline | Fine-tuned | Delta |
|---|---|---|---|
| JSON validity | 70.0% | 100.0% | **+30.0%** |
| Intent accuracy | 76.0% | 97.3% | **+21.3%** |
| Gatekeeper acc. | 74.0% | 97.3% | **+23.3%** |
| Slot F1 | 0.700 | 0.960 | **+0.260** |
| Hallucination rt. | 12.0% | 0.7% | **−11.3%** |

*(Numbers above are PRD targets — actual results may vary)*

---

### Script Numbering (Week 2 onwards)

| Script | Purpose |
|---|---|
| `scripts/03_baseline_eval.py` | Eval any model on test.jsonl (baseline + fine-tuned) |
| `scripts/04_compare_results.py` | Generate before/after comparison report |
| `scripts/05_train_qwen.py` | Fine-tune Qwen-2.5-7B via Unsloth (Track A) |
| `scripts/06_train_llama.py` | Fine-tune Llama-3.2-3B via Unsloth (Track B) |
| `scripts/07_evaluate_models.py` | Week 3: full 5-model benchmark + LLM judge |

*Week 4 will be documented here as it is completed.*

---

## Week 3 — Multi-Model Benchmark ✅ Complete

**Status:** Complete. 5-model benchmark run locally. Results in `results/`.

**Goal:** Evaluate all 5 model variants on 150 held-out test rows with automated metrics, LLM-as-a-Judge scoring, and bootstrap 95% confidence intervals — producing the full "before / after / teacher gap" story for the portfolio.

**Runs on:** Local Mac — DeepSeek-V3 teacher eval via API (~150 calls), DeepSeek-V3 judge via API (~750 calls). No GPU required.

**How to run:**
```bash
python scripts/07_evaluate_models.py          # full run (~10 min)
python scripts/07_evaluate_models.py --skip-teacher --skip-judge  # metrics only (instant)
```

### 5 Models Evaluated

| Model Slug | Description |
|---|---|
| `base_qwen` | Qwen-2.5-7B untrained base |
| `ft_qwen` | Qwen-2.5-7B fine-tuned (Track A, LoRA r=16) |
| `base_llama` | Llama-3.2-3B untrained base |
| `ft_llama` | Llama-3.2-3B fine-tuned (Track B, LoRA r=8) |
| `teacher_deepseek` | DeepSeek-V3 — the knowledge distillation teacher |

### LLM-as-a-Judge

**DeepSeek-V3** (`deepseek-chat`) was used as the judge (not Claude Haiku) — at ~$0.11 for 750 calls it was the cheapest option, and since it generated the golden training data it has perfect familiarity with the expected schema. Runs async with concurrency=10 via the existing `DeepSeekClient`. Checkpointed — safe to interrupt and resume.

### Bootstrap Confidence Intervals

1,000 resamples per metric per model. Confirms that improvements are statistically real (non-overlapping CIs between base and fine-tuned models).

### Actual Results (Run 1)

| Model | JSON Valid | Intent Acc. | GK Acc. | Slot F1 | Halluc. | Judge Score |
|---|---|---|---|---|---|---|
| Base Qwen-2.5-7B | 99.3% | 61.3% | 33.3% | 91.8% | 2.0% | 3.08 / 5 |
| **FT Qwen-2.5-7B (Track A)** | 96.7% | **86.0%** | **62.0%** | 92.0% | 2.7% | **3.94 / 5** |
| Base Llama-3.2-3B | 94.7% | 24.0% | 26.0% | 86.3% | 4.7% | 2.39 / 5 |
| **FT Llama-3.2-3B (Track B)** | **98.0%** | **74.0%** | **58.0%** | **91.6%** | **0.7%** | **3.69 / 5** |
| DeepSeek-V3 (teacher) | 89.3% | 81.3% | 73.3% | 85.8% | 2.0% | 4.07 / 5 |

**Key deltas (Run 1):**
- Qwen: Intent +24.7pp, GK +28.7pp, Slot F1 +0.002, Halluc +0.7pp (already low at 2%)
- Llama: Intent +50.0pp, GK +32.0pp, Slot F1 +0.052, Halluc −4.0pp

**Note:** Gatekeeper accuracy on both fine-tuned models came in below PRD targets (62% vs ≥97% for Track A; 58% vs ≥92% for Track B). This is the primary motivation for the planned second training run — likely due to class imbalance in the training data or prompt engineering that needs refinement.

### Output Files

| File | Contents |
|---|---|
| `results/teacher_results_deepseek.json` | Teacher eval — 150 rows + summary |
| `results/eval_results.json` | Unified 5-model × 150-row data + judge scores |
| `results/eval_summary.json` | Aggregated metrics + bootstrap 95% CIs |
| `results/cross_comparison_table.md` | Portfolio money table (Δ vs base, teacher gap) |
| `results/comparison_report.json` | Machine-readable before/after delta data |
| `results/comparison_report.md` | Human-readable before/after Markdown table |

---

## Week 4 — Dashboard + Portfolio Packaging ✅ Complete

**Status:** Dashboard live at https://ecommerce-return-gatekeeper.vercel.app. Neon Postgres seeded. GitHub repo public.

**Goal:** Build a Next.js 14 analytics dashboard that visualises all 5 benchmark models,
exposes an interactive row inspector, cost calculator, and triage router demo.

**Pages (all live):**
| Route | What it shows |
|---|---|
| `/` | KPI Summary — grouped bar chart (5 metrics × 5 models) + LLM judge scores with 95% CI |
| `/rows` | Row Inspector — list of 150 test seed_ids |
| `/rows/[id]` | Row detail — 5 model outputs side-by-side with ground truth |
| `/cost` | Cost Calculator — slider (volume) → monthly cost table + ΔROI vs GPT-4o |
| `/router` | Triage Router Demo — type a message → instant routing decision |

**Stack:**
- Next.js 14 (App Router, `force-dynamic` server components)
- Recharts v3 (bar charts with `isAnimationActive={false}` for SSR compatibility)
- Neon Postgres via `@vercel/postgres` (`POSTGRES_URL` env var in Vercel settings)
- Deployed on Vercel (project: `ecommerce-return-gatekeeper`, team: `yasir-aiportfolio`)

**How to run locally:**
```bash
# Local dev (needs POSTGRES_URL in dashboard/.env.local)
cd dashboard && npm install && npm run dev
```

**How to re-seed the database (if results change):**
```bash
pip install psycopg2-binary
# Use the unpooled connection string from Vercel Storage → Neon tab
POSTGRES_URL="postgres://..." python scripts/seed_db.py
```

**Database schema:** `dashboard/src/lib/schema.sql` — two tables: `model_summaries` (5 rows, one per model) and `eval_rows` (750 rows, 5 models × 150 test rows).

**Important technical decisions:**
- `next.config.ts` is NOT supported by Next.js 14.2 — must use `next.config.mjs`
- All Recharts `Bar` components use `isAnimationActive={false}` to prevent blank charts in SSR
- `mounted` state guard in `ModelComparisonChart.tsx` prevents hydration mismatch

**HF Hub:** Fine-tuned adapter weights were not uploaded — the RunPod pod was terminated before upload. Weights will be uploaded after the second training run.
```bash
huggingface-cli upload <username>/qwen-2.5-7b-ecommerce-gk ./output/qwen-2.5-7b-ecommerce-gk
huggingface-cli upload <username>/llama-3.2-3b-ecommerce-gk ./output/llama-3.2-3b-ecommerce-gk
```

---

## Planned: Second Training Run

**Motivation:** First-run gatekeeper accuracy (62% for Qwen Track A, 58% for Llama Track B) significantly underperformed PRD targets (≥97% and ≥92%). Intent accuracy gains were strong (+24.7pp and +50.0pp), but gatekeeper decision-making needs improvement.

**Goal:** Re-train both models, compare Run 2 results against Run 1, update `eval_summary.json` and re-seed the dashboard.

**Likely improvements to investigate before Run 2:**
- Check class balance of `gatekeeper_status` in `train.jsonl` — if one class dominates, the model learns to always predict it
- Review whether the system prompt during fine-tuning inference matches the training format exactly
- Consider adding more `ESCALATE_TO_HUMAN` and `REQUEST_EVIDENCE` examples if `APPROVE_AUTOMATED` is overrepresented

**RunPod execution order (Run 2):**
```bash
# 1. (Optional) Re-generate dataset if class balance fix is needed
#    Or just re-run training with adjusted hyperparameters

# 2. Fine-tune
python scripts/05_train_qwen.py
python scripts/06_train_llama.py

# 3. Evaluate fine-tuned models
python scripts/03_baseline_eval.py --model Qwen/Qwen2.5-7B-Instruct --adapter-path output/qwen-2.5-7b-ecommerce-gk --load-in-4bit
python scripts/03_baseline_eval.py --model meta-llama/Llama-3.2-3B-Instruct --adapter-path output/llama-3.2-3b-ecommerce-gk --load-in-4bit

# 4. Locally: run full 5-model benchmark + judge
python scripts/07_evaluate_models.py

# 5. Re-seed dashboard
POSTGRES_URL="..." python scripts/seed_db.py
```
