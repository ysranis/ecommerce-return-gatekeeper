# E-commerce Return Gatekeeper — Full 5-Model Benchmark

_Generated: 2026-07-15T09:33:36.033146+00:00_

## Before vs. After Fine-Tuning + Teacher Comparison

| Metric | Base Qwen | FT Qwen (A) | Δ(A) | Base Llama | FT Llama (B) | Δ(B) | DeepSeek-V3 (Teacher) |
|---|---|---|---|---|---|---|---|
| JSON validity | 99.3% | 96.7% | **-2.7%** | 94.7% | 98.0% | **+3.3%** | 89.3% |
| Intent accuracy | 61.3% | 86.0% | **+24.7%** | 24.0% | 74.0% | **+50.0%** | 81.3% |
| Gatekeeper acc. | 33.3% | 62.0% | **+28.7%** | 26.0% | 58.0% | **+32.0%** | 73.3% |
| Slot F1 | 0.918 | 0.920 | **+0.002** | 0.863 | 0.916 | **+0.052** | 0.858 |
| Hallucination rt. | 2.0% | 2.7% | **+0.7%** | 4.7% | 0.7% | **-4.0%** | 2.0% |

## 95% Bootstrap Confidence Intervals

| Metric | FT Qwen CI | FT Llama CI | Teacher CI |
|---|---|---|---|
| JSON validity | [93.3%, 99.3%] | [95.3%, 100.0%] | [84.7%, 94.0%] |
| Intent accuracy | [80.0%, 91.3%] | [67.3%, 80.7%] | [75.3%, 87.3%] |
| Gatekeeper acc. | [54.0%, 70.0%] | [50.0%, 66.0%] | [66.0%, 80.7%] |
| Slot F1 | [0.878, 0.958] | [0.874, 0.952] | [0.800, 0.908] |
| Hallucination rt. | [0.7%, 5.3%] | [0.0%, 2.0%] | [0.0%, 4.7%] |