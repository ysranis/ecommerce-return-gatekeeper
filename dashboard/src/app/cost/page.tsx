import CostCalculator from "@/components/CostCalculator";

export default function CostPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Cost Calculator</h1>
        <p className="text-gray-400 mt-1">
          Compare monthly costs across all model tiers at your dispute volume.
          Fine-tuned self-hosted models deliver frontier-level accuracy at 3&ndash;83&times; lower cost.
        </p>
      </div>
      <CostCalculator />
    </div>
  );
}
