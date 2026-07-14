"use client";
import { useState, useEffect } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ErrorBar,
} from "recharts";
import type { ModelSummary } from "@/lib/db";
import { MODEL_LABELS } from "@/lib/constants";

const METRIC_COLORS: Record<string, string> = {
  json_validity: "#60a5fa",
  intent_accuracy: "#34d399",
  gatekeeper_accuracy: "#f59e0b",
  slot_f1: "#a78bfa",
  hallucination_rate: "#f87171",
};

type Props = { summaries: ModelSummary[] };

export default function ModelComparisonChart({ summaries }: Props) {
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  const data = summaries.map((s) => ({
    name: MODEL_LABELS[s.slug] ?? s.slug,
    "JSON validity": +(s.json_validity_mean * 100).toFixed(1),
    "Intent acc.": +(s.intent_accuracy_mean * 100).toFixed(1),
    "GK acc.": +(s.gatekeeper_accuracy_mean * 100).toFixed(1),
    "Slot F1": +(s.slot_f1_mean * 100).toFixed(1),
    "Hallucination": +(s.hallucination_rate_mean * 100).toFixed(1),
  }));

  const judgeData = summaries.map((s) => ({
    name: MODEL_LABELS[s.slug] ?? s.slug,
    "Judge score": +s.judge_score_mean.toFixed(2),
    errorY: [
      +(s.judge_score_mean - s.judge_score_ci_lower).toFixed(2),
      +(s.judge_score_ci_upper - s.judge_score_mean).toFixed(2),
    ],
  }));

  if (!mounted) return <div style={{ height: 620 }} />;

  return (
    <div className="space-y-10">
      <div>
        <h2 className="text-lg font-semibold mb-4 text-gray-300">
          5-Metric Comparison (% or × 100)
        </h2>
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={data} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="name" tick={{ fill: "#9ca3af", fontSize: 12 }} />
            <YAxis domain={[0, 100]} tick={{ fill: "#9ca3af", fontSize: 12 }} />
            <Tooltip
              contentStyle={{ background: "#1f2937", border: "1px solid #374151" }}
              labelStyle={{ color: "#f3f4f6" }}
            />
            <Legend wrapperStyle={{ color: "#9ca3af" }} />
            {Object.entries(METRIC_COLORS).map(([key, color]) => (
              <Bar key={key} isAnimationActive={false} dataKey={
                key === "json_validity" ? "JSON validity" :
                key === "intent_accuracy" ? "Intent acc." :
                key === "gatekeeper_accuracy" ? "GK acc." :
                key === "slot_f1" ? "Slot F1" : "Hallucination"
              } fill={color} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div>
        <h2 className="text-lg font-semibold mb-4 text-gray-300">
          LLM Judge Score (1–5, with 95% CI)
        </h2>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={judgeData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="name" tick={{ fill: "#9ca3af", fontSize: 12 }} />
            <YAxis domain={[0, 5]} tick={{ fill: "#9ca3af", fontSize: 12 }} />
            <Tooltip
              contentStyle={{ background: "#1f2937", border: "1px solid #374151" }}
              labelStyle={{ color: "#f3f4f6" }}
            />
            <Bar dataKey="Judge score" fill="#818cf8" isAnimationActive={false}>
              <ErrorBar dataKey="errorY" width={4} strokeWidth={2} stroke="#a5b4fc" />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
