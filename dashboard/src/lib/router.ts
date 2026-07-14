const HIGH_COMPLEXITY_INTENTS = new Set(["get_refund", "complaint", "check_refund_policy"]);
const HIGH_EMOTION_MARKERS = new Set(["HIGH_EMOTION", "OFFENSIVE", "FRUSTRATED"]);

export function detectIntent(message: string): string {
  const m = message.toLowerCase();
  if (
    m.includes("refund") &&
    (m.includes("where") || m.includes("track") || m.includes("status"))
  )
    return "track_refund";
  if (m.includes("refund") || m.includes("money back") || m.includes("return"))
    return "get_refund";
  if (m.includes("cancel")) return "cancel_order";
  if (
    m.includes("complaint") ||
    m.includes("terrible") ||
    m.includes("awful") ||
    m.includes("worst")
  )
    return "complaint";
  if (
    m.includes("policy") ||
    m.includes("rules") ||
    m.includes("how long") ||
    m.includes("allowed")
  )
    return "check_refund_policy";
  return "cancel_order";
}

export function detectEmotions(message: string): string[] {
  const m = message.toLowerCase();
  const emotions: string[] = [];
  if (m.includes("!!!") || m.includes("furious") || m.includes("outraged"))
    emotions.push("HIGH_EMOTION");
  if (
    m.includes("disgusting") ||
    m.includes("stupid") ||
    m.includes("idiot")
  )
    emotions.push("OFFENSIVE");
  if (
    m.includes("frustrated") ||
    m.includes("angry") ||
    m.includes("disappointed")
  )
    emotions.push("FRUSTRATED");
  return emotions;
}

export type RouteResult = {
  intent: string;
  emotions: string[];
  track: "track_a" | "track_b";
  model: string;
  reason: string;
};

export function routeDispute(message: string): RouteResult {
  const intent = detectIntent(message);
  const emotions = detectEmotions(message);
  const track =
    HIGH_COMPLEXITY_INTENTS.has(intent) ||
    emotions.some((e) => HIGH_EMOTION_MARKERS.has(e))
      ? "track_a"
      : "track_b";
  return {
    intent,
    emotions,
    track,
    model:
      track === "track_a"
        ? "Qwen-2.5-7B-Instruct (Track A)"
        : "Llama-3.2-3B-Instruct (Track B)",
    reason: HIGH_COMPLEXITY_INTENTS.has(intent)
      ? `Intent "${intent}" requires high-accuracy Track A model`
      : emotions.length > 0
      ? `Emotion markers [${emotions.join(", ")}] escalate to Track A`
      : `Simple procedural intent "${intent}" → Track B (fast + cheap)`,
  };
}
