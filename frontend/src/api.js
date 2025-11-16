const API_BASE = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

/**
 * Streams tokens from POST /api/chat/stream
 * The backend sends Server-Sent-Event style chunks: "data: {...}\n\n"
 */
export async function* streamAnswer(query) {
  const resp = await fetch(`${API_BASE}/api/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });
  if (!resp.ok || !resp.body) {
    throw new Error(`HTTP ${resp.status}`);
  }
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });

    let idx;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const raw = buf.slice(0, idx).trim();
      buf = buf.slice(idx + 2);
      if (!raw.startsWith("data:")) continue;
      const data = raw.slice(5).trim(); // after "data:"
      if (data === "[DONE]") return;
      try {
        const obj = JSON.parse(data);
        // This is correct: it yields the token string, 
        // or "" if it's a citation object
        yield obj.token ?? ""; 
      } catch (e) {
        // --- THIS IS THE FIX ---
        // If parsing fails, log it and yield nothing
        // instead of the broken raw text.
        console.error("Failed to parse stream chunk:", data, e);
        yield "";
      }
    }
  }
}