import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")  # 384 dims, English

def embed(texts: list[str]) -> np.ndarray:
    return model.encode(texts, normalize_embeddings=False)

def cosine_similarity(a, b):
    # Ensure inputs are treated as NumPy arrays
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    
    # Calculate the dot product
    dot_product = np.dot(a, b)
    
    # Calculate the L2 Euclidean norm (magnitude) of each vector
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    # Avoid division by zero if one of the vectors is a zero-vector
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    # Return the cosine similarity score
    return dot_product / (norm_a * norm_b)

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

embedd_docs = embed(docs)
print("Embedding shape:", embedd_docs[0].shape)

print("Embedding dtype:", embedd_docs[0].dtype)

def search(query, k=3):
    quer_embed = embed([query])[0]
    similarities = [cosine_similarity(quer_embed, doc_embed) for doc_embed in embedd_docs]
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    return [(docs[i], similarities[i]) for i in top_k_indices]


print(cosine_similarity(embedd_docs[5], embedd_docs[6])) 

print(cosine_similarity(embedd_docs[0], embedd_docs[1]))

# docsSim = search("quantum chromodynamics")

# print("Top similar documents:")
# for doc, sim in docsSim:
#     print(f"Document: {doc}, Similarity: {sim:.4f}")