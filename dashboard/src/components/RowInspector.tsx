"use client";
import type { EvalRow } from "@/lib/db";

const MODEL_LABELS: Record<string, string> = {
  base_qwen: "Base Qwen",
  ft_qwen: "FT Qwen (A)",
  base_llama: "Base Llama",
  ft_llama: "FT Llama (B)",
  teacher_deepseek: "DeepSeek-V3",
};

function Badge({ ok, label }: { ok: boolean | null; label: string }) {
  if (ok === null) return <span className="text-gray-500">{label}</span>;
  return (
    <span className={ok ? "text-green-400" : "text-red-400"}>
      {ok ? "✓" : "✗"} {label}
    </span>
  );
}

function ModelCard({ row }: { row: EvalRow }) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 space-y-3 min-w-0">
      <div className="font-semibold text-white text-sm">
        {MODEL_LABELS[row.model_slug] ?? row.model_slug}
      </div>
      <div className="text-xs space-y-1">
        <div className="flex gap-2">
          <Badge ok={row.intent_correct} label={row.pred_intent_action ?? "—"} />
        </div>
        <div className="flex gap-2">
          <Badge ok={row.gatekeeper_correct} label={row.pred_gatekeeper_status ?? "—"} />
        </div>
        {row.judge_score !== null && (
          <div className="text-yellow-400">Judge: {row.judge_score}/5</div>
        )}
        {row.hallucinated_slots && (
          <div className="text-red-400">Hallucinated slots</div>
        )}
      </div>
      {row.pred_intent_action && (
        <div className="text-xs text-gray-500 space-y-1">
          <div>order_id: <span className="text-gray-300">{row.pred_order_id ?? "null"}</span></div>
          <div>invoice_id: <span className="text-gray-300">{row.pred_invoice_id ?? "null"}</span></div>
        </div>
      )}
      {row.judge_reason && (
        <div className="text-xs text-gray-400 italic border-t border-gray-800 pt-2">
          {row.judge_reason}
        </div>
      )}
    </div>
  );
}

type Props = { rows: EvalRow[] };

export default function RowInspector({ rows }: Props) {
  if (!rows.length) return <div className="text-gray-400">No data found.</div>;

  const first = rows[0];

  return (
    <div className="space-y-6">
      {/* Customer message */}
      <div className="bg-gray-900 border border-blue-800 rounded-xl p-4">
        <div className="text-xs text-blue-400 font-semibold mb-2">CUSTOMER MESSAGE</div>
        <div className="text-gray-200 text-sm leading-relaxed">{first.synthetic_message}</div>
      </div>

      {/* Ground truth */}
      <div className="bg-gray-900 border border-green-800 rounded-xl p-4 text-sm">
        <div className="text-xs text-green-400 font-semibold mb-2">GROUND TRUTH</div>
        <div className="grid grid-cols-2 gap-2 text-gray-300">
          <div>Intent: <span className="text-white">{first.gt_intent_action}</span></div>
          <div>Decision: <span className="text-white">{first.gt_gatekeeper_status}</span></div>
          <div>Order ID: <span className="text-white">{first.gt_order_id ?? "null"}</span></div>
          <div>Invoice ID: <span className="text-white">{first.gt_invoice_id ?? "null"}</span></div>
        </div>
      </div>

      {/* Model outputs */}
      <div>
        <div className="text-xs text-gray-500 font-semibold mb-3">MODEL OUTPUTS</div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-3">
          {rows.map((r) => (
            <ModelCard key={r.model_slug} row={r} />
          ))}
        </div>
      </div>
    </div>
  );
}
