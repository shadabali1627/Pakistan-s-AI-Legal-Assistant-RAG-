// Use your new secure domain (Note the /api at the end)
const API_BASE = "http://127.0.0.1:8000";

/**
 * Streams tokens from POST /api/chat/stream
 * The backend sends Server-Sent-Event style chunks: "data: {...}\n\n"
 *
 * @param {string} query The new user query
 * @param {Array<{role: string, content: string}>} history The previous chat messages
 */
export async function* streamAnswer(query, history = [], signal = null, user_email = null, session_id = null) {
  const resp = await fetch(`${API_BASE}/api/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    signal, // Pass the aborter signal
    // 1. Send both the query and the history
    body: JSON.stringify({ query, history, k: 6, user_email, session_id }), // k is a default, you can make it dynamic
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

        // 2. Yield the typed event directly
        if (obj.type) {
          yield obj; // { type: 'status'|'content'|'citations'|'error', data: ... }
        }
        // Legacy fallback (shouldn't be needed but good for safety)
        else if (obj.token) {
          yield { type: "content", data: obj.token };
        } else if (obj.citations) {
          yield { type: "citations", data: obj.citations };
        }
      } catch (e) {
        console.error("Failed to parse stream chunk:", data, e);
        // yield nothing on parse error
      }
    }
  }
}