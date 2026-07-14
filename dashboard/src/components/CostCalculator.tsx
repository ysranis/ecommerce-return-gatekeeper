"use client";
import { useState } from "react";

const MODELS = [
  { name: "GPT-4o (frontier baseline)", costPerDispute: 0.005, color: "text-red-400" },
  { name: "DeepSeek-V3 (teacher)", costPerDispute: 0.00027, color: "text-yellow-400" },
  { name: "Qwen-2.5-7B (Track A, self-hosted)", costPerDispute: 0.00015, color: "text-blue-400" },
  { name: "Llama-3.2-3B (Track B, self-hosted)", costPerDispute: 0.00006, color: "text-green-400" },
];

function fmt(n: number): string {
  return n >= 1000 ? `$${n.toLocaleString("en-US", { maximumFractionDigits: 0 })}` : `$${n.toFixed(2)}`;
}

export default function CostCalculator() {
  const [volume, setVolume] = useState(10000);

  const gpt4oCost = volume * MODELS[0].costPerDispute;

  return (
    <div className="space-y-8">
      <div>
        <label className="block text-sm text-gray-400 mb-2">
          Monthly dispute volume: <span className="text-white font-semibold">{volume.toLocaleString()}</span>
        </label>
        <input
          type="range"
          min={1000}
          max={100000}
          step={1000}
          value={volume}
          onChange={(e) => setVolume(Number(e.target.value))}
          className="w-full accent-blue-500"
        />
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>1,000</span><span>100,000</span>
        </div>
      </div>

      <table className="w-full text-sm border-collapse">
        <thead>
          <tr className="border-b border-gray-800">
            <th className="text-left py-3 text-gray-400 font-medium">Model</th>
            <th className="text-right py-3 text-gray-400 font-medium">Cost/dispute</th>
            <th className="text-right py-3 text-gray-400 font-medium">Monthly cost</th>
            <th className="text-right py-3 text-gray-400 font-medium">vs GPT-4o</th>
          </tr>
        </thead>
        <tbody>
          {MODELS.map((m) => {
            const monthly = volume * m.costPerDispute;
            const saving = gpt4oCost - monthly;
            return (
              <tr key={m.name} className="border-b border-gray-800/50">
                <td className={`py-3 font-medium ${m.color}`}>{m.name}</td>
                <td className="py-3 text-right text-gray-300">${m.costPerDispute.toFixed(5)}</td>
                <td className="py-3 text-right text-gray-100 font-semibold">{fmt(monthly)}</td>
                <td className={`py-3 text-right font-semibold ${saving > 0 ? "text-green-400" : "text-gray-500"}`}>
                  {saving > 0 ? `−${fmt(saving)}` : "—"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>

      <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 text-sm text-gray-400">
        Assumes ~1,000 tokens per dispute (input + output). Self-hosted costs use A10G GPU at $0.44/hr.
        Costs are estimates and will vary by provider and usage patterns.
      </div>
    </div>
  );
}
