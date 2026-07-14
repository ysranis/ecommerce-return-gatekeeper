import { getModelSummaries } from "@/lib/db";
import ModelComparisonChart from "@/components/ModelComparisonChart";

export const dynamic = "force-dynamic";

export default async function HomePage() {
  const summaries = await getModelSummaries();

  const ftQwen = summaries.find((s) => s.slug === "ft_qwen");
  const ftLlama = summaries.find((s) => s.slug === "ft_llama");
  const baseQwen = summaries.find((s) => s.slug === "base_qwen");

  return (
    <div className="space-y-10">
      <div>
        <h1 className="text-3xl font-bold text-white">E-commerce Return Gatekeeper</h1>
        <p className="text-gray-400 mt-2">
          Multi-tiered LLM dispute arbitration — knowledge distillation from DeepSeek-V3 into
          Qwen-2.5-7B (Track A) and Llama-3.2-3B (Track B).
        </p>
      </div>

      {/* Hero KPI cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
          <div className="text-3xl font-bold text-blue-400">
            {ftQwen ? `+${((ftQwen.intent_accuracy_mean - (baseQwen?.intent_accuracy_mean ?? 0)) * 100).toFixed(1)}pp` : "—"}
          </div>
          <div className="text-sm text-gray-400 mt-1">Intent accuracy gain — Qwen (A)</div>
        </div>
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
          <div className="text-3xl font-bold text-green-400">
            {ftQwen ? ftQwen.judge_score_mean.toFixed(2) : "—"} / 5
          </div>
          <div className="text-sm text-gray-400 mt-1">LLM Judge score — FT Qwen</div>
        </div>
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
          <div className="text-3xl font-bold text-purple-400">
            {ftLlama ? `+${(ftLlama.judge_score_mean - (summaries.find(s => s.slug === "base_llama")?.judge_score_mean ?? 0)).toFixed(2)}` : "—"}
          </div>
          <div className="text-sm text-gray-400 mt-1">LLM Judge gain — Llama (B)</div>
        </div>
      </div>

      <ModelComparisonChart summaries={summaries} />
    </div>
  );
}
