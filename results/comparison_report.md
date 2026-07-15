# E-commerce Return Gatekeeper — Before vs. After Fine-Tuning

_Generated: 2026-07-15T12:57:48.679465+00:00_

## Qwen-2.5-7B (Track A — Accuracy-Optimized)

**Base model:** `Qwen/Qwen2.5-7B-Instruct`
**Adapter:** `output/qwen-2.5-7b-ecommerce-gk-v2`

| Metric | Baseline | Fine-tuned | Delta |
|---|---|---|---|
| JSON validity | 99.3% | 100.0% | **+0.7%** |
| Intent accuracy | 61.3% | 86.0% | **+24.7%** |
| Gatekeeper acc. | 33.3% | 64.0% | **+30.7%** |
| Slot F1 | 0.918 | 0.958 | **+0.040** |
| Hallucination rt. | 2.0% | 0.7% | **-1.3%** |

## Llama-3.2-3B (Track B — Speed-Optimized)

**Base model:** `meta-llama/Llama-3.2-3B-Instruct`
**Adapter:** `/root/output/llama-3.2-3b-ecommerce-gk`

| Metric | Baseline | Fine-tuned | Delta |
|---|---|---|---|
| JSON validity | 94.7% | 98.0% | **+3.3%** |
| Intent accuracy | 24.0% | 74.0% | **+50.0%** |
| Gatekeeper acc. | 26.0% | 58.0% | **+32.0%** |
| Slot F1 | 0.863 | 0.916 | **+0.052** |
| Hallucination rt. | 4.7% | 0.7% | **-4.0%** |

## Key Takeaways

- **Qwen-2.5-7B (Track A — Accuracy-Optimized)**: 5/5 metrics improved after fine-tuning
- **Llama-3.2-3B (Track B — Speed-Optimized)**: 5/5 metrics improved after fine-tuning
