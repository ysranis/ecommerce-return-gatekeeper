import TriageRouterDemo from "@/components/TriageRouterDemo";

export default function RouterPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Triage Router Demo</h1>
        <p className="text-gray-400 mt-1">
          The Dynamic Triage Router assigns each dispute to the most cost-effective model.
          Track A (Qwen-7B) handles complex or emotionally charged disputes.
          Track B (Llama-3B) handles simple procedural tasks.
        </p>
        <p className="text-xs text-gray-500 mt-2">
          This demo uses rule-based intent + emotion detection — the same logic as{" "}
          <code className="bg-gray-800 px-1 rounded">router/triage_router.py</code>.
          No live inference is performed.
        </p>
      </div>
      <TriageRouterDemo />
    </div>
  );
}
