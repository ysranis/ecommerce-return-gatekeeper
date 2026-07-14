# E-commerce Return Gatekeeper — Before vs. After Fine-Tuning

_Generated: 2026-07-14T14:05:39.995527+00:00_

## Qwen-2.5-7B (Track A — Accuracy-Optimized)

**Base model:** `Qwen/Qwen2.5-7B-Instruct`
**Adapter:** `/root/output/qwen-2.5-7b-ecommerce-gk`

| Metric | Baseline | Fine-tuned | Delta |
|---|---|---|---|
| JSON validity | 99.3% | 96.7% | **-2.7%** |
| Intent accuracy | 61.3% | 86.0% | **+24.7%** |
| Gatekeeper acc. | 33.3% | 62.0% | **+28.7%** |
| Slot F1 | 0.918 | 0.920 | **+0.002** |
| Hallucination rt. | 2.0% | 2.7% | **+0.7%** |

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

- **Qwen-2.5-7B (Track A — Accuracy-Optimized)**: 3/5 metrics improved after fine-tuning
- **Llama-3.2-3B (Track B — Speed-Optimized)**: 5/5 metrics improved after fine-tuning
