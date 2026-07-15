# E-commerce Return Gatekeeper

A 4-week ML portfolio project demonstrating **knowledge distillation + QLoRA fine-tuning** for automated e-commerce dispute arbitration.

**Live dashboard:** [ecommerce-return-gatekeeper.vercel.app](https://ecommerce-return-gatekeeper.vercel.app)

---

## What it does

A customer submits a return/refund dispute. The system:

1. **Routes** the dispute via a rule-based Triage Router — complex/emotional cases go to Track A (Qwen-7B), simple procedural ones to Track B (Llama-3B)
2. **Arbitrates** using a fine-tuned LLM that outputs a structured JSON decision: intent, extracted slots, policy evaluation, gatekeeper status, and a user-facing response
3. **Benchmarks** all 5 models (2 base + 2 fine-tuned + teacher) on 150 held-out test rows across 5 metrics, scored by an LLM judge

---

## Architecture

```
DeepSeek-V3 (teacher)
    │
    ▼ knowledge distillation (2000 synthetic disputes + targeted RE supplement)
    ├─ Qwen-2.5-7B  ──── QLoRA r=32 (Run 2) ──── Track A (accuracy)
    └─ Llama-3.2-3B ──── QLoRA r=16 (Run 3) ──── Track B (speed/cost)
          │
          ▼
    Dynamic Triage Router ──── rule-based intent + emotion detection
          │
          ▼
    Next.js Dashboard ──── Vercel + Neon Postgres
```

---

## Results

| Model | Intent Acc. | GK Acc. | Slot F1 | Judge Score |
|---|---|---|---|---|
| Base Qwen-2.5-7B | 61.3% | 33.3% | 91.8% | 3.08 / 5 |
| **FT Qwen-2.5-7B (A)** | **86.0%** | **62.0%** | **92.0%** | **3.94 / 5** |
| Base Llama-3.2-3B | 24.0% | 26.0% | 86.3% | 2.39 / 5 |
| **FT Llama-3.2-3B (B)** | **74.0%** | **58.0%** | **91.6%** | **3.69 / 5** |
| DeepSeek-V3 (teacher) | 81.3% | 73.3% | 85.8% | 4.07 / 5 |

**Complete fine tunning analysis:** [Wandb Report]([https://api.wandb.ai/links/ysranis-/sqy5spnp])

Fine-tuning lifted Qwen intent accuracy by **+24.7pp** and Llama by **+50pp**, at 3–83× lower cost than frontier API models.

> **Run 2 note:** Qwen was retrained with a targeted REQUEST_EVIDENCE supplement (train_v2, 1,411 rows, r=32) which improved gatekeeper accuracy from 62% → 64% and intent accuracy from 74% → 86%. Llama best result is kept from Run 1 — Run 2/3 experiments showed diminishing returns for the 3B model at higher RE ratios.

---

## Model adapters (Hugging Face Hub)

| Model | Repo | Run |
|---|---|---|
| Qwen-2.5-7B fine-tuned | [yasiranis/qwen-2.5-7b-ecommerce-gk-v2](https://huggingface.co/yasiranis/qwen-2.5-7b-ecommerce-gk-v2) | Run 2 |
| Llama-3.2-3B fine-tuned | [yasiranis/llama-3.2-3b-ecommerce-gk-v3](https://huggingface.co/yasiranis/llama-3.2-3b-ecommerce-gk-v3) | Run 3 |

LoRA adapters only — load on top of the respective base models via `peft`.

---

## Dashboard pages

| Page | Description |
|---|---|
| [KPI Summary](https://ecommerce-return-gatekeeper.vercel.app/) | Hero metrics + 5-model bar chart + LLM judge scores with 95% CI |
| [Row Inspector](https://ecommerce-return-gatekeeper.vercel.app/rows) | Browse all 150 test rows, compare 5 model outputs side-by-side |
| [Cost Calculator](https://ecommerce-return-gatekeeper.vercel.app/cost) | Interactive volume slider — monthly cost vs GPT-4o at any scale |
| [Router Demo](https://ecommerce-return-gatekeeper.vercel.app/router) | Live triage routing with intent + emotion detection |

---

## Project structure

```
scripts/
  01_generate_dataset.py   # Bitext ingestion + DeepSeek-V3 synthetic generation
  02_label_dataset.py      # Golden labeling + train/val/test split
  03_baseline_eval.py      # Eval any model on test.jsonl (base or fine-tuned)
  04_compare_results.py    # Before/after comparison report (JSON + MD)
  05_train_qwen.py         # Qwen-2.5-7B QLoRA fine-tuning via Unsloth (Run 2: r=32)
  06_train_llama.py        # Llama-3.2-3B QLoRA fine-tuning via Unsloth (Run 3: r=16)
  07_evaluate_models.py    # 5-model benchmark + LLM-as-Judge scoring
  08_seed_db.py            # Seed Neon Postgres from results JSON
  09_generate_request_evidence.py  # Targeted RE class supplement generation

router/
  triage_router.py         # Rule-based Dynamic Triage Router

dashboard/                 # Next.js 14 app (deployed on Vercel)
  src/app/                 # Pages: /, /rows, /cost, /router
  src/components/          # ModelComparisonChart, RowInspector, CostCalculator, TriageRouterDemo
  src/lib/                 # db.ts (Neon Postgres), router.ts, constants.ts

results/
  eval_summary.json        # Aggregated metrics with 95% CI per model
  eval_results.json        # Per-row predictions across all 5 models
  comparison_report.md     # Before/after fine-tuning comparison

tests/                     # 35 passing unit tests
```

---

## Stack

| Layer | Technology |
|---|---|
| Dataset | Bitext customer support seed + DeepSeek-V3 distillation |
| Fine-tuning | Unsloth + SFTTrainer (trl), RunPod A10G |
| Eval | transformers pipeline, Claude Haiku 4.5 as LLM judge |
| Tracking | Weights & Biases |
| Dashboard | Next.js 14, Recharts, Tailwind CSS |
| Database | Neon Postgres (via Vercel Storage) |
| Deployment | Vercel |

---

## Running locally

```bash
# Python environment (scripts + tests)
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pytest -v

# Dashboard
cd dashboard
npm install
npm run dev
# Set POSTGRES_URL in .env.local
```

---

## Cost

Total project cost: ~$5–6 (RunPod training + DeepSeek-V3 API + LLM judge calls).
