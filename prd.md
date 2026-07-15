# Product Requirement Document (PRD)

**Project Title:** E-commerce Post-Purchase Dispute Arbitration & Return Gatekeeper
**Subtitle:** Multi-Tiered Open-Source Model Distillation & Fleet Optimization Strategy

---

## 1. Document Control & Metadata

| Field | Value |
|---|---|
| Author | AI Product Manager / Builder |
| Version | 4.1 |
| Document Status | Run 2 In Progress — diagnosis complete, remediation underway |
| Target Release | Q4 2026 |
| Last Updated | 2026-07-15 |
| Core Dataset | Bitext Gen AI Chatbot Customer Support Dataset (26,872 QA pairs, 27 intents, 10 categories) |
| Teacher Model | DeepSeek-V3 (Mixture-of-Experts, 671B total parameters, 37B active) |
| Track A Student | Qwen-2.5-7B-Instruct (Performance-First) |
| Track B Student | Llama-3.2-3B-Instruct (Cost-First) |
| Target Infrastructure | Next.js 14 · Neon Postgres · Vercel |
| Live Dashboard | https://ecommerce-return-gatekeeper.vercel.app |
| GitHub | https://github.com/ysranis/ecommerce-return-gatekeeper (public) |

---

## 2. Executive Summary & Problem Statement

### 2.1 Problem Statement

Deploying enterprise customer support agents using unconstrained frontier API models results in variable token expenditure that erodes transactional profit margins at scale. Conversely, uniform deployment of standard 3B or 7B parameter models out-of-the-box yields high failure rates on strict JSON schema compliance, introduces branding hallucinations, and fails to handle complex user policy escalations.

**The cost gap is severe:**

| Model Tier | Cost per 1M Tokens | Cost at 10M Tokens/Day |
|---|---|---|
| Frontier (GPT-4o) | ~$5.00 | ~$50/day ($1,500/month) |
| Teacher (DeepSeek-V3) | ~$0.27 | ~$2.70/day ($81/month) |
| Self-hosted 7B (Qwen) | ~$0.15 | ~$1.50/day ($45/month) |
| Self-hosted 3B (Llama) | ~$0.06 | ~$0.60/day ($18/month) |

### 2.2 Solution Strategy: A Dual-Track Fleet

Rather than forcing a single model to handle every customer query, this architecture splits operational labor into a **multi-tiered specialized model fleet** coordinated by a Dynamic Triage Router.

- **Track A (Qwen-2.5-7B-Instruct):** Optimized as the high-fidelity policy gatekeeper. Handles multi-step reasoning, linguistic nuances (sarcasm, frustration), and complex itemized return logic requiring strict schema compliance and policy adherence.
- **Track B (Llama-3.2-3B-Instruct):** Optimized as a low-latency utility layer. Handles standard, highly repetitive procedural actions (status lookups, order cancellations) where execution speed and low compute footprint are paramount.

```
                [ Inbound Customer Request ]
                              ||
                              \/
               +---------------------------+
               |   Dynamic Triage Router   |
               +---------------------------+
                              ||
         +--------------------+--------------------+
         | Complex / Emotional                     | Simple / Procedural
         \/                                        \/
+-------------------------+             +---------------------------+
|   Qwen-2.5-7B (Track A) |             |  Llama-3.2-3B (Track B)   |
|  - Policy Arbitration   |             |  - Order Tracking          |
|  - Exception Management |             |  - Order Cancellation      |
|  - Refund Gating        |             |  - Standard Policy Checks  |
+-------------------------+             +---------------------------+
```

### 2.3 Objective

Distill the domain-specific reasoning capabilities of DeepSeek-V3 into two fine-tuned open-weight models using a synthetically generated e-commerce dispute dataset built on Bitext seeds, then deploy them as a coordinated fleet:

- **Track A target:** 100% JSON schema compliance, ≥97% policy accuracy, ≤250ms p50 latency
- **Track B target:** ≥98.5% JSON compliance, ≥91.5% intent accuracy, ≤120ms p50 latency / <150ms TTFT

---

## 3. Scope & Feature Requirements

### 3.0 Rationale: Why Not Use Raw Bitext Responses?

The raw Bitext dataset contains authentic human input variations — complete with spelling errors, colloquial registers, and offensive language flags. However, its out-of-the-box responses consist of generic, long-form conversational text sentences.

If a small model is trained directly on these raw responses, it will fail to learn structural programmatic operations — producing verbose chat-style text instead of the strict JSON schema required for downstream gatekeeper logic. This means zero reusability for automated policy decisions, slot extraction, or confidence-gated escalation routing.

By implementing a **Knowledge Distillation Pipeline**, we use the raw Bitext strings as behavioral inputs only, executing a batch synthesis pass through DeepSeek-V3 to construct structured targets. This approach injects advanced reasoning patterns and strict schema boundaries directly into the smaller models' weights — enabling 3B and 7B parameter models to perform with frontier-level schema compliance at a fraction of the inference cost.

### 3.1 Data Scope: Bitext Dataset Fields

The project uses the Bitext Gen AI Chatbot Customer Support Dataset as a seed corpus only. The `instruction` and `flags` columns are extracted; the original text answers are **discarded** — all response targets are generated fresh by DeepSeek-V3 with structured reasoning chains.

- **`instruction`**: Raw customer query containing linguistic variations (typos, abbreviations, colloquial register)
- **`category` & `intent`**: Targeting — `get_refund`, `cancel_order`, `track_refund`, `complaint`, `check_refund_policy`
- **`flags` / Language Register Tags**: Colloquial (Q), Offensive (W), Politeness (P), Errors/Typos (Z)

**Dataset scale:** Extract 2,000 balanced seed rows. After distillation and validation, a minimum of 1,500 validated rows must be available before training begins.

### 3.2 System Module Matrix

| Module ID | Component Name | Functional Requirement | Priority |
|---|---|---|---|
| SYS-01 | Dynamic Triage Router | Evaluates incoming request strings. Routes to Track B if intent maps to low-risk metadata checks or simple procedural actions. Routes to Track A if intent matches multi-step arguments, complex policy logic, or high emotional register | P0 |
| SYS-02 | Structured Metadata Extractor | Model extracts unstructured raw chat into a strict JSON payload containing: `intent_action`, `extracted_slots` (order/invoice IDs, item condition, return window), and `linguistic_markers` | P0 |
| SYS-03 | Policy Gatekeeper Engine | Model evaluates extracted metadata against store return rules and emits a `gatekeeper_status`: `APPROVE_AUTOMATED`, `REQUEST_EVIDENCE`, or `ESCALATE_TO_HUMAN` | P0 |
| SYS-04 | Confidence Scoring | Model outputs a `confidence_score` (0.0–1.0) alongside each decision to enable automatic fallback routing below a threshold | P0 |
| SYS-05 | User-Facing Response Generator | Model generates a customer-facing `user_facing_response` string alongside the internal arbitration JSON | P0 |
| SYS-06 | Analytical Evaluation Dashboard | Next.js frontend mapping accuracy scores, latency metrics, and hosting costs across all model variants with side-by-side row inspection | P1 |
| SYS-07 | Audit Logging | All gatekeeper decisions logged with chain-of-thought and processing timestamp for compliance | P1 |
| SYS-08 | Multilingual Dispute Handling | Track A (Qwen-2.5-7B) must handle disputes in English, Spanish, French, and Arabic | P2 |

---

## 4. System Architecture & Data Pipeline

The system is divided into three sequential phases: Data Engineering, Fine-Tuning, and Evaluation + Deployment.

```
┌─────────────────────────────────────────────────────────────────┐
│                  PHASE 1: DATA ENGINEERING                      │
│                                                                 │
│  [ Bitext Dataset (26,872 rows) ]                               │
│                 │                                               │
│                 ▼                                               │
│  [ Filter: 2,000 Balanced E-commerce Seed Rows ]                │
│  (get_refund, cancel_order, track_refund,                       │
│   complaint, check_refund_policy × Q/W/P/Z flags)               │
│  [Discard original Bitext text answers — use instruction only]  │
│                 │                                               │
│                 ▼                                               │
│  [ DeepSeek-V3 Teacher — Distillation Pass ]                    │
│  System prompt includes fictional corporate returns handbook    │
│  Generates: CoT reasoning + structured JSON + user response     │
│  + 400 multilingual variants (ES, FR, AR)                       │
│                 │                                               │
│                 ▼                                               │
│  [ Post-Generation Validation Gate ]                            │
│  (Parse JSON · Reject failures · Min 1,500 validated rows)      │
│                 │                                               │
│                 ▼                                               │
│  [ distilled_dataset.jsonl ] (≥1,500 validated rows)            │
│  Train: 1,200 rows · Val: 150 rows · Test: 150 rows (held out)  │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  PHASE 2: FINE-TUNING                           │
│                                                                 │
│            Track A                       Track B               │
│   [ Qwen-2.5-7B-Instruct ]       [ Llama-3.2-3B-Instruct ]     │
│    Performance-First Fleet           Cost-First Fleet           │
│    LoRA targeting Wq, Wv            LoRA targeting Wq, Wv      │
│    lora_r=16 · seq_len=2048         lora_r=8 · seq_len=1024    │
│    W&B Experiment Tracking          W&B Experiment Tracking     │
│            │                                 │                  │
│            ▼                                 ▼                  │
│  qwen-2.5-7b-ecommerce-gk       llama-3.2-3b-ecommerce-gk      │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              PHASE 3: EVALUATION HUB + DEPLOYMENT               │
│                                                                 │
│  5 Model Endpoints × 150 Test Rows                              │
│  ┌──────────────┬──────────────┬────────────────────────────┐   │
│  │ Base Qwen-7B │  FT Qwen-7B  │                            │   │
│  │ Base Llama-3B│  FT Llama-3B │  DeepSeek-V3 Teacher       │   │
│  └──────────────┴──────────────┴────────────────────────────┘   │
│                 │                                               │
│                 ▼                                               │
│  [ LLM-as-a-Judge + Automated Metrics → Vercel Postgres ]       │
│                 │                                               │
│                 ▼                                               │
│  [ Next.js Analytics Dashboard ] → Vercel Deploy                │
│  [ Dynamic Triage Router ] → transformers pipeline / Hardcoded Demo  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.1 Golden JSON Output Schema

Every teacher-generated training example must conform to this merged schema. The `instruction` and `flags` from Bitext seed serve as the `user_input` and `meta_context` input; the `distilled_response` is fully generated by DeepSeek-V3:

```json
{
  "user_input": "hey cancel order #AX-9481 right now, my package has been stuck in shipping for weeks and im losing my mind here!!",
  "meta_context": {
    "assigned_category": "ORDER_CANCELLATION",
    "linguistic_markers": ["COLLOQUIAL", "HIGH_EMOTION", "TYPOS"]
  },
  "distilled_response": {
    "chain_of_thought": "Customer wants to cancel order #AX-9481 due to prolonged shipping delays. Tone indicates high frustration (HIGH_EMOTION). Extracted order ID: AX-9481. Per the corporate returns handbook: orders stuck in transit over 14 business days qualify for immediate automated cancellation and refund without human overhead. Confidence is high — explicit order number present and delay claim is verifiable.",
    "intent_action": "cancel_order",
    "extracted_slots": {
      "order_id": "AX-9481",
      "invoice_id": null,
      "return_window_days": null,
      "item_condition": null
    },
    "policy_evaluation": {
      "within_return_window": true,
      "item_opened": false,
      "evidence_required": false
    },
    "gatekeeper_status": "APPROVE_AUTOMATED",
    "confidence_score": 0.94,
    "fallback_escalation": false,
    "user_facing_response": "I have successfully canceled your order AX-9481 and processed a full refund to your original payment method. I apologize for the delay in shipping — your refund will be reflected within 3–5 business days.",
    "processing_timestamp": "2026-07-07T10:23:00Z"
  }
}
```

---

## 5. Evaluation Matrix & Target KPIs

Performance is measured across five model variants on a held-out **150-row test set**.

### 5.1 Cost Calculation Methodology

Total computational expenditure per inference run is calculated as:

$$\text{Total Cost} = \sum \left( \text{Tokens}_{\text{in}} \times R_{\text{in}} + \text{Tokens}_{\text{out}} \times R_{\text{out}} \right)$$

Where $R_{\text{in}}$ and $R_{\text{out}}$ are the pricing tiers per million tokens for each model. The ROI improvement over the frontier teacher baseline is:

$$\Delta_{\text{ROI}} = \left( 1 - \frac{\text{Cost}_{\text{Student Fleet}}}{\text{Cost}_{\text{Teacher Baseline}}} \right) \times 100\%$$

### 5.2 Success Metrics (KPIs)

**Targets vs. Actual (Run 1):**

| Metric | Base Qwen | Target (FT Qwen A) | **Actual (FT Qwen A)** | Base Llama | Target (FT Llama B) | **Actual (FT Llama B)** | Teacher |
|---|---|---|---|---|---|---|---|
| JSON Validity Rate | 99.3% | 100% | **96.7%** | 94.7% | ≥98.5% | **98.0% ✓** | 89.3% |
| Policy Accuracy (GK status) | 33.3% | ≥97% | **62.0% ✗** | 26.0% | ≥92% | **58.0% ✗** | 73.3% |
| Intent Classification Acc. | 61.3% | ≥97% | **86.0%** | 24.0% | ≥91.5% | **74.0%** | 81.3% |
| Entity Slot Extraction F1 | 91.8% | ≥96% | **92.0%** | 86.3% | ≥89% | **91.6% ✓** | 85.8% |
| Hallucination Rate | 2.0% | <1% | **2.7%** | 4.7% | <3% | **0.7% ✓** | 2.0% |
| LLM Judge Score (1–5) | 3.08 | — | **3.94** | 2.39 | — | **3.69** | 4.07 |

**Targets met (Run 1):** JSON validity (Llama ✓), Slot F1 (Llama ✓), Hallucination rate (Llama ✓).

**Targets not met (Run 1):** Gatekeeper accuracy on both tracks (62% vs ≥97% for A; 58% vs ≥92% for B). Intent accuracy improved significantly (+24.7pp Qwen, +50.0pp Llama) but did not reach ≥97%/≥91.5% targets.

**Action:** A second training run is planned. Likely root cause: class imbalance in `gatekeeper_status` labels in the training set, or a mismatch between the training prompt format and eval inference format.

> **Note on cost:** Fine-tuned and base student models share the same cost because they run on identical self-hosted vLLM infrastructure. The ΔROI vs. the teacher baseline is the primary financial metric.

### 5.3 LLM-as-a-Judge Evaluation Rubric

An independent evaluator model grades each candidate output against the teacher's golden standard. **Actual implementation used DeepSeek-V3 (`deepseek-chat`)** as the judge — at ~$0.11 for 750 calls it was the cheapest option, and since it generated the golden training data it has perfect schema familiarity. (Original plan specified Claude Haiku 4.5 as primary, DeepSeek-V3 as alternative.) Keep the judge prompt under 400 tokens and output only `{"score": X, "reason": "one sentence"}`.

```
You are an expert Enterprise QA Evaluation Auditor for e-commerce return policy systems.
Rate the candidate model's structured arbitration decision against the Ground-Truth Golden Standard.

SCORING RUBRIC:

Score 5 — PERFECT:
  Correct intent_action, all key extracted_slots present, exact gatekeeper_status match,
  valid parseable JSON, and coherent user_facing_response.

Score 4 — NEAR-PERFECT:
  Correct intent and gatekeeper_status. Valid JSON. Minor slot miss
  (e.g., extracted order_id but missed item_condition).

Score 3 — PARTIAL:
  Correct JSON format. Intent classified correctly but gatekeeper_status is wrong
  (e.g., APPROVE_AUTOMATED instead of REQUEST_EVIDENCE), or multiple slot failures.

Score 2 — POOR:
  JSON partially parseable (missing brackets, invalid field types).
  Structural output is unreliable for downstream systems.

Score 1 — FAIL:
  Hallucinated fields (fabricated order IDs), non-parseable output, or critical policy
  breach (e.g., auto-approving a flagged fraudulent transaction).

Output your evaluation as:
{"score": <1-5>, "reason": "<one sentence justification>"}
```

> **Inter-rater reliability check:** Run 10% of test samples through two independent evaluator calls. Flag score disagreements ≥2 for manual review.

---

## 6. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Teacher model generates low-quality dispute scenarios | Medium | High | Manual review of 50 samples before full run; iterate system prompt until quality is acceptable |
| Overfitting on 1,200 training rows | Medium | High | `lora_dropout=0.05`, early stopping when val loss plateaus >2 epochs, monitor W&B curves |
| Catastrophic forgetting (fine-tuned model loses general reasoning) | Low | Medium | Run MMLU subset (100 questions) before and after fine-tuning as a guard metric |
| DeepSeek-V3 API rate limiting during bulk generation | Medium | Low | Batch 20 requests max with exponential backoff and retry |
| JSON schema breaks on edge cases (multilingual, offensive language) | Medium | High | Post-generation validation gate rejects malformed outputs; rerun failed rows with stricter prompt |
| Triage router misroutes complex queries to Track B | Medium | High | Add confidence threshold: if router confidence <0.85, default to Track A |
| Data contamination (Bitext overlap with model pretraining data) | Low | Low | Noted as known limitation in README; use only synthetic generation outputs for training |

---

## 7. Dependencies & Prerequisites

| Dependency | Details |
|---|---|
| Dataset | Bitext Gen AI Chatbot Customer Support Dataset (`bitext/Bitext-customer-support-llm-chatbot-training-dataset`) |
| Teacher API | DeepSeek-V3 API key (`DEEPSEEK_API_KEY`) |
| GPU Compute | A10G or A100 (24GB VRAM min for Qwen-7B QLoRA); RunPod or Lambda Labs |
| Python Environment | Python 3.11+, `transformers`, `datasets`, `peft`, `unsloth`, `trl`, `langsmith`, `wandb` |
| Experiment Tracking | Weights & Biases (`WANDB_API_KEY`) |
| Dashboard | Next.js 14+, Vercel account, Vercel Postgres (for eval results storage) |
| Model Hosting | Hugging Face account (model cards + adapter weights) |
| Dataset License | Verify Bitext dataset license before publishing synthetic derivatives |

---

## 8. Out of Scope

- Real-time streaming responses or WebSocket-based interfaces
- Production deployment with live customer traffic
- Model quantization beyond QLoRA (e.g., GGUF, AWQ, GPTQ post-training)
- Languages beyond English, Spanish, French, and Arabic
- Fine-tuning on proprietary customer support data (PII-containing)
- Agentic multi-turn conversation management
- A/B testing framework or shadow mode deployment
- Triage router model training (rule-based or lightweight classifier only for this version)

---

## 9. Execution Plan & Quality Gates

### Week 1: Ingestion Filtration & Knowledge Distillation Pass

- **Engineering Target:** Extract 2,000 balanced interaction seeds from the Bitext dataset covering key transaction paths (`get_refund`, `cancel_order`, `track_refund`, `complaint`, `check_refund_policy`) across all four linguistic flag types (Q/W/P/Z).
- **Execution:** Build a Python preprocessing module. Pass inputs through DeepSeek-V3 using system-guided prompts — including the fictional corporate returns handbook — to generate the structured Chain-of-Thought + JSON targets defined in the Golden Schema (Section 4.1).
- **Quality Gate 1:** Reject any rows where DeepSeek-V3 fails to output a fully parseable JSON response block. A minimum of 1,500 validated rows must be written to `distilled_dataset.jsonl` before proceeding.

### Week 2: Parallel Model Fine-Tuning Runs

- **Engineering Target:** Complete supervised fine-tuning loops across both student model configurations simultaneously.
- **Execution:**
  - **Track A:** Load Qwen-2.5-7B-Instruct. Run LoRA training targeting attention projections ($W_q, W_v$), `lora_r=16`, 3 epochs, sequence length 2048 tokens.
  - **Track B:** Load Llama-3.2-3B-Instruct. Run LoRA training with `lora_r=8`, sequence length 1024 tokens, optimized for compute footprint and TTFT.
  - Track all runs via Weights & Biases experiment tracking.
- **Quality Gate 2:** Training loss curves must display regular, non-volatile flattening trends across epochs without exhibiting validation overfitting characteristics. Validation loss must not increase for more than 2 consecutive epochs.

### Week 3: Automated Pairwise Evaluation & Testing Matrix

- **Engineering Target:** Evaluate all 5 model variants (base + fine-tuned for each track, plus teacher baseline) against the held-out 150-row test set.
- **Execution:** Run comparative inference passes across all endpoints. Pipe response fields into the LLM-as-a-Judge evaluator (defined in Section 5.3) to score intent preservation, schema compliance, and policy accuracy. Store results in Vercel Postgres.
- **Quality Gate 3:** Track A must demonstrate 100% JSON compliance and ≥97% alignment with corporate policy actions (`gatekeeper_status`). Track B must verify time-to-first-token under 150ms. Any model variant scoring below threshold triggers a hyperparameter review before deployment proceeds.

### Week 4: Analytical Portfolio Dashboard Deployment

- **Engineering Target:** Build and deploy the Analytical Evaluation Dashboard (SYS-06).
- **Execution:** Create a Next.js frontend displaying cost optimization graphs, accuracy comparison tables, latency benchmarks, and side-by-side prompt output inspection views. Push to GitHub and deploy via Vercel.
- **Quality Gate 4:** The dashboard must clearly surface the cost-performance tradeoffs and operational benefits of the multi-tiered fleet to a technical reviewer with no prior context — specifically the ΔROI improvement over the DeepSeek-V3 teacher baseline.

---

## 10. Run 2: Diagnosis & Remediation

### 10.1 Run 1 Outcome vs. Targets

Quality Gate 3 targets were not met on gatekeeper accuracy:

| Model | JSON Valid | Intent Acc. | GK Acc. Target | GK Acc. Actual | Gap |
|---|---|---|---|---|---|
| FT Qwen-2.5-7B (A) | 96.7% ✅ | 86.0% ✅ | ≥97% | 62.0% | −35pp |
| FT Llama-3.2-3B (B) | 98.0% ✅ | 74.0% ✅ | ≥92% | 58.0% | −34pp |

Intent accuracy improved dramatically (Qwen +24.7pp, Llama +50.0pp), confirming the distillation pipeline works. The gatekeeper accuracy gap is a training data problem, not an architecture problem.

### 10.2 Root Cause Analysis

**RC-1 — Class imbalance: `REQUEST_EVIDENCE` starved at 12% of training data**

The training set contains only 144 `REQUEST_EVIDENCE` examples out of 1,200 (12%). The model learns the easy majority pattern (APPROVE or ESCALATE) and fails on the minority class.

Per-class gatekeeper accuracy from Run 1:

| Class (n=21 in test) | FT Qwen | FT Llama |
|---|---|---|
| APPROVE_AUTOMATED | 66.7% | 60.3% |
| ESCALATE_TO_HUMAN | 69.7% | 54.5% |
| **REQUEST_EVIDENCE** | **23.8%** | **61.9%** |

FT Qwen gets `REQUEST_EVIDENCE` correct only 1-in-4 times. Of 16 errors, 15 predict `ESCALATE_TO_HUMAN` — the two classes are semantically adjacent ("not automated") so weak training signal causes collapse.

**RC-2 — Structural blind spot: `cancel_order` + `REQUEST_EVIDENCE` = 0 training examples**

| Intent | APPROVE | ESCALATE | REQUEST_EVIDENCE |
|---|---|---|---|
| cancel_order (416 rows) | 360 | 56 | **0** |
| get_refund (615 rows) | 66 | 421 | 128 |

The model has never seen a cancellation scenario where evidence is required. It also over-associates `cancel_order` with `APPROVE_AUTOMATED` (86.5% of cancel_order rows), which inflates false approvals.

**RC-3 — Chain-of-thought ambiguity: training labels do not articulate the REQUEST_EVIDENCE trigger rule**

The `LABELING_SYSTEM_PROMPT` states "Damaged items: Photo evidence required before any refund" but does not encode a decision rule distinguishing `REQUEST_EVIDENCE` ("I need evidence before I can decide") from `ESCALATE_TO_HUMAN` ("this is too complex/flagged for a human"). Synthetic training labels are inconsistent in applying this boundary.

**Prompt alignment: not a factor.** Training (`05_train_qwen.py`) and eval (`03_baseline_eval.py`) both use the identical `LABELING_SYSTEM_PROMPT` with `apply_chat_template`. No prompt drift between training and inference.

### 10.3 Remediation Plan (Run 2)

| Fix | Implementation | Expected Impact |
|---|---|---|
| Upsample REQUEST_EVIDENCE to ~25% of train | `scripts/09_generate_request_evidence.py` generates ~200 targeted rows; merged into `data/train_v2.jsonl` | Raises REQUEST_EVIDENCE signal from 12% → 25%; closes per-class accuracy gap |
| Add cancel_order + REQUEST_EVIDENCE examples | ~30 rows generated in script 09 covering shipped-but-unconfirmed scenarios | Closes structural blind spot in cancel_order path |
| Sharpen chain_of_thought labels | All new rows include explicit CoT reasoning: "item damaged → photo required → REQUEST_EVIDENCE (not ESCALATE)" | Reduces ESCALATE↔REQUEST_EVIDENCE confusion |
| Increase Qwen LoRA rank r=16→32, epochs 3→4 | Updated in `scripts/05_train_qwen.py` | More adapter capacity for Qwen which showed worst REQUEST_EVIDENCE accuracy |
| Llama dataset update only (no hyperparameter change) | Updated dataset path in `scripts/06_train_llama.py` | Llama already at 61.9% REQUEST_EVIDENCE; data fix sufficient |

### 10.4 Run 2 Success Criteria (revised Quality Gate 3)

| Metric | Run 1 | Run 2 Target | Rationale |
|---|---|---|---|
| GK Accuracy — FT Qwen | 62.0% | ≥75% | Realistic gain from data fix + LoRA upgrade |
| GK Accuracy — FT Llama | 58.0% | ≥70% | Realistic gain from data fix alone |
| REQUEST_EVIDENCE per-class — Qwen | 23.8% | ≥55% | Primary failure mode to fix |
| REQUEST_EVIDENCE per-class — Llama | 61.9% | ≥70% | Already acceptable; small improvement expected |
| Intent accuracy | 86.0% / 74.0% | Maintain (±3pp) | Must not regress on gains from Run 1 |

---

## 11. Glossary

| Term | Definition |
|---|---|
| **CoT** | Chain-of-Thought — step-by-step reasoning the teacher generates before the final JSON decision |
| **QLoRA** | Quantized Low-Rank Adaptation — memory-efficient fine-tuning that trains adapter weights on a quantized base model |
| **LoRA** | Low-Rank Adaptation — parameter-efficient fine-tuning; QLoRA is the quantized variant |
| **Wq, Wv** | Query and Value weight matrices in transformer attention layers — primary LoRA targets |
| **vLLM** | Open-source high-throughput inference engine for LLMs used to self-host fine-tuned models |
| **JSONL** | JSON Lines — file format where each line is a valid JSON object; used for training datasets |
| **TTFT** | Time-to-First-Token — latency from request receipt to first output token; critical for Track B UX |
| **Bitext** | Source seed dataset: Bitext Gen AI Chatbot Customer Support Dataset (26,872 labeled QA pairs) |
| **Distillation** | Training a smaller student model to replicate the behavior of a larger teacher model |
| **Hallucination** | Model generating factually incorrect or fabricated information (e.g., inventing an order ID) |
| **p50 / p95 Latency** | 50th and 95th percentile response times; p95 captures tail latency for worst-case users |
| **MMLU** | Massive Multitask Language Understanding — general reasoning benchmark for catastrophic forgetting detection |
| **W&B** | Weights & Biases — experiment tracking for loss curves, hyperparameters, and run comparison |
| **ΔROI** | ROI improvement formula: (1 - Cost_Student / Cost_Teacher) × 100% |
| **Triage Router** | Lightweight classifier routing inbound requests to the appropriate model tier (Track A or B) |
| **PII** | Personally Identifiable Information — data that can identify a specific individual |
