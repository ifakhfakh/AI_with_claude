import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("shop_docs")   # raises if it doesn't exist
print("chunks stored:", collection.count())


def chroma_search(query, k=3, section=None):
    query_vec = model.encode([query], normalize_embeddings=True)[0].tolist()

    kwargs = {}
    if section is not None:
        kwargs["where"] = {"section": section}

    res = collection.query(
        query_embeddings=[query_vec],
        n_results=k,
        **kwargs,
    )
    return [
        (1 - dist, doc, meta)
        for doc, dist, meta in zip(
            res["documents"][0], res["distances"][0], res["metadatas"][0]
        )
    ]


for q in ["How long do I have to return an item?", "What isn't covered by the warranty?"]:
    print(f"\n{q}")
    for score, text, meta in chroma_search(q):
        print(f"  {score:.4f} | {meta['section']} | {text[:60]!r}")