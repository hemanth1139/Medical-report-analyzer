from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def load_index(facts_path="data/medical_facts.txt"):
    with open(facts_path, "r") as f:
        facts = [line.strip() for line in f if line.strip()]
    embeddings = MODEL.encode(facts)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, facts

def retrieve(query, index, facts, top_k=2):
    query_vec = MODEL.encode([query])
    _, indices = index.search(np.array(query_vec), top_k)
    return [facts[i] for i in indices[0]]
