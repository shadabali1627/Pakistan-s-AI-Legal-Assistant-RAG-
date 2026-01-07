
from backend.rag_pipeline import _classify_intent

queries = [
    "tell about human rights",
    "tell me about human rights",
    "what are my rights",
    "human rights in pakistan",
    "hello",
    "how are you"
]

print("--- Testing Intent Classifier ---")
for q in queries:
    intent = _classify_intent(q, "gemma-2-9b-it")
    print(f"Query: '{q}' -> Intent: {intent}")
