"use client";
import { useState } from "react";
import { routeDispute, type RouteResult } from "@/lib/router";

const EXAMPLES = [
  "I want a refund for my broken headphones, order AX-4832. This is absolutely outrageous!!!",
  "Hey can you tell me where my refund is? It's been 5 days, order AX-4927.",
  "I need to cancel order AX-1234 please.",
  "What is your return policy for opened items?",
];

export default function TriageRouterDemo() {
  const [message, setMessage] = useState("");
  const [result, setResult] = useState<RouteResult | null>(null);

  function handleRoute() {
    if (!message.trim()) return;
    setResult(routeDispute(message));
  }

  return (
    <div className="space-y-6 max-w-2xl">
      <div className="space-y-3">
        <label className="block text-sm text-gray-400">Type a customer dispute message:</label>
        <textarea
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          rows={4}
          className="w-full bg-gray-900 border border-gray-700 rounded-xl px-4 py-3 text-gray-100 text-sm focus:outline-none focus:border-blue-500 resize-none"
          placeholder="e.g. I want my money back for the damaged item I received last week..."
        />
        <button
          onClick={handleRoute}
          disabled={!message.trim()}
          className="px-5 py-2 bg-blue-600 hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed rounded-lg text-sm font-medium text-white transition-colors"
        >
          Route this dispute
        </button>
      </div>

      <div className="space-y-2">
        <div className="text-xs text-gray-500 font-semibold">EXAMPLES</div>
        <div className="space-y-2">
          {EXAMPLES.map((ex) => (
            <button
              key={ex}
              onClick={() => { setMessage(ex); setResult(null); }}
              className="block w-full text-left text-xs text-gray-400 hover:text-gray-200 bg-gray-900/50 border border-gray-800 rounded-lg px-3 py-2 transition-colors"
            >
              {ex}
            </button>
          ))}
        </div>
      </div>

      {result && (
        <div className={`border rounded-xl p-5 space-y-3 ${
          result.track === "track_a"
            ? "bg-blue-950/40 border-blue-700"
            : "bg-green-950/40 border-green-700"
        }`}>
          <div className="flex items-center gap-3">
            <span className={`text-lg font-bold ${result.track === "track_a" ? "text-blue-300" : "text-green-300"}`}>
              {result.track === "track_a" ? "Track A" : "Track B"}
            </span>
            <span className="text-gray-300 text-sm">{result.model}</span>
          </div>
          <div className="text-sm text-gray-400">{result.reason}</div>
          <div className="flex flex-wrap gap-2 text-xs">
            <span className="bg-gray-800 text-gray-300 rounded px-2 py-1">intent: {result.intent}</span>
            {result.emotions.map((e) => (
              <span key={e} className="bg-red-900/50 text-red-300 rounded px-2 py-1">{e}</span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
