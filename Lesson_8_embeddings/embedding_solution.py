import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed(texts: list[str]) -> np.ndarray:
    return model.encode(texts, normalize_embeddings=False)

def cosine_similarity(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

docs = [
    "How do I reset my password?",
    "I forgot my login credentials and can't sign in.",
    "The cat sat on the warm windowsill.",
    "Our refund policy allows returns within 30 days.",
    "Python lists are mutable, tuples are not.",
    "I love this product.",
    "I hate this product.",
    "A kitten was lounging in the sun by the window.",
]

doc_vecs = embed(docs)

# Task 1: shape, dtype, norm
print("shape:", doc_vecs[0].shape, "| dtype:", doc_vecs[0].dtype)
print("L2 norm of first vector:", np.linalg.norm(doc_vecs[0]))

# Task 2/3: loop version, returns (score, text) as specified
def search_loop(query, k=3):
    q = embed([query])[0]
    sims = [cosine_similarity(q, d) for d in doc_vecs]
    top = np.argsort(sims)[::-1][:k]
    return [(sims[i], docs[i]) for i in top]

# Task 5: normalize, then dot product == cosine
def normalize(m):
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / norms

doc_norm = normalize(doc_vecs)

a, b = doc_vecs[0], doc_vecs[1]
print("cosine:", cosine_similarity(a, b))
print("dot of normalized:", float(np.dot(doc_norm[0], doc_norm[1])))
print("equal to 6 decimals:",
      round(cosine_similarity(a, b), 6) == round(float(np.dot(doc_norm[0], doc_norm[1])), 6))

# Optional: loop-free search, one matrix multiply
def search_fast(query, k=3):
    q = normalize(embed([query]))[0]
    sims = doc_norm @ q            # shape (n_docs,), all similarities at once
    top = np.argsort(sims)[::-1][:k]
    return [(float(sims[i]), docs[i]) for i in top]

# Task 4: queries (plus the "no relevant doc" test)
queries = [
    "I can't log in",
    "feline resting near a window",
    "tell me about immutability",
    "I adore this product",
    "quantum chromodynamics",
]
for q in queries:
    loop_res = search_loop(q)
    fast_res = search_fast(q)
    same_ranking = [t for _, t in loop_res] == [t for _, t in fast_res]
    print(f"\nQuery: {q}  (loop and fast agree: {same_ranking})")
    for score, text in fast_res:
        print(f"  {score:.4f}  {text}")
    print(f"  top score: {fast_res[0][0]:.4f}")

# Task 4b: the love/hate comparison
print("\ndocs 0 vs 1 (paraphrase):", round(cosine_similarity(doc_vecs[0], doc_vecs[1]), 4))
print("docs 5 vs 6 (love/hate): ", round(cosine_similarity(doc_vecs[5], doc_vecs[6]), 4))