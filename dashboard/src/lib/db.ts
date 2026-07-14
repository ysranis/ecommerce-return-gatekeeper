import { sql } from "@vercel/postgres";

export type ModelSummary = {
  slug: string;
  label: string;
  json_validity_mean: number;
  json_validity_ci_lower: number;
  json_validity_ci_upper: number;
  intent_accuracy_mean: number;
  intent_accuracy_ci_lower: number;
  intent_accuracy_ci_upper: number;
  gatekeeper_accuracy_mean: number;
  gatekeeper_accuracy_ci_lower: number;
  gatekeeper_accuracy_ci_upper: number;
  slot_f1_mean: number;
  slot_f1_ci_lower: number;
  slot_f1_ci_upper: number;
  hallucination_rate_mean: number;
  hallucination_rate_ci_lower: number;
  hallucination_rate_ci_upper: number;
  judge_score_mean: number;
  judge_score_ci_lower: number;
  judge_score_ci_upper: number;
};

export type EvalRow = {
  seed_id: string;
  model_slug: string;
  synthetic_message: string | null;
  gt_intent_action: string | null;
  gt_gatekeeper_status: string | null;
  gt_order_id: string | null;
  gt_invoice_id: string | null;
  json_valid: boolean | null;
  pred_intent_action: string | null;
  pred_gatekeeper_status: string | null;
  pred_order_id: string | null;
  pred_invoice_id: string | null;
  intent_correct: boolean | null;
  gatekeeper_correct: boolean | null;
  slot_f1: number | null;
  hallucinated_slots: boolean | null;
  judge_score: number | null;
  judge_reason: string | null;
};

export async function getModelSummaries(): Promise<ModelSummary[]> {
  const { rows } = await sql<ModelSummary>`
    SELECT * FROM model_summaries
    ORDER BY CASE slug
      WHEN 'base_qwen' THEN 1
      WHEN 'ft_qwen' THEN 2
      WHEN 'base_llama' THEN 3
      WHEN 'ft_llama' THEN 4
      WHEN 'teacher_deepseek' THEN 5
      ELSE 6
    END
  `;
  return rows;
}

export async function getEvalRowIds(): Promise<string[]> {
  const { rows } = await sql<{ seed_id: string }>`
    SELECT DISTINCT seed_id FROM eval_rows ORDER BY seed_id
  `;
  return rows.map((r) => r.seed_id);
}

export async function getEvalRowById(seedId: string): Promise<EvalRow[]> {
  const { rows } = await sql<EvalRow>`
    SELECT * FROM eval_rows WHERE seed_id = ${seedId}
    ORDER BY CASE model_slug
      WHEN 'base_qwen' THEN 1
      WHEN 'ft_qwen' THEN 2
      WHEN 'base_llama' THEN 3
      WHEN 'ft_llama' THEN 4
      WHEN 'teacher_deepseek' THEN 5
      ELSE 6
    END
  `;
  return rows;
}
