
from backend.rag_pipeline import retrieve_mmr, retrieve_with_scores, _format_docs, _classify_intent, _condense_query_with_history
import sys

def debug_rag(query):
    output = []
    output.append(f"--- Debugging Query: '{query}' ---")
    
    # 1. Intent
    intent = _classify_intent(query, "gemma-2-9b-it")
    output.append(f"Intent Class: {intent}")
    
    # 2. Retrieval
    output.append("Retrieving docs...")
    filtered_docs = retrieve_mmr(query, k=8)
    if not filtered_docs:
        output.append("MMR returned empty. Falling back to Similarity with Scores...")
        scored = retrieve_with_scores(query, k=6)
        filtered_docs = [d for d, s in scored if 0.0 <= s <= 1.0 and s >= 0.20]
    
    output.append(f"Found {len(filtered_docs)} docs.")
    
    # 3. Context Dump
    output.append("\n--- EXTRACTED CONTEXT ---")
    for i, doc in enumerate(filtered_docs):
        meta = doc.metadata
        source = meta.get("source", "unknown")
        page = meta.get("page", "?")
        output.append(f"\n[Doc {i+1}] Source: {source} | Page: {page}")
        content = doc.page_content.replace("\n", " ").strip()
        output.append(f"Content: {content}")
        
    with open("extracted_text.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(output))

if __name__ == "__main__":
    debug_rag("tell me about human rights in simple words according to pakistani law")
