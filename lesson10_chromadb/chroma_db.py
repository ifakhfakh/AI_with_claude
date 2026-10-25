import re
import numpy as np
from sentence_transformers import SentenceTransformer

DOC = """Returns and Refunds

Our refund policy allows returns within 30 days of delivery. Items must be unused and in their original packaging. Refunds are issued to the original payment method within 5 to 7 business days after we receive the item.

Digital products, gift cards, and personalized items cannot be returned. If a digital product is defective, contact support and we will replace it.

Shipping

We ship to the United States, Canada, and the European Union. Standard shipping takes 5 to 10 business days. Express shipping takes 2 to 3 business days and costs extra. Orders over 75 dollars qualify for free standard shipping.

International orders may be subject to customs duties and taxes, which are the responsibility of the customer. We cannot predict these charges in advance.

Warranty

All electronic products include a one-year limited warranty from the date of purchase. The warranty covers manufacturing defects only. It does not cover accidental damage, water damage, or normal wear and tear. To make a claim, keep your receipt and contact support with photos of the defect.

Accounts

You can reset your password from the login page by clicking Forgot Password. If you no longer have access to your email address, contact support with proof of identity. Accounts inactive for 24 months may be deleted after a notice by email."""

model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------------------------------------------- Task 1
doc_tokens = model.tokenizer.encode(DOC)  # includes 2 special tokens
print("max_seq_length:", model.max_seq_length)
print("tokens in DOC:", len(doc_tokens))
visible_ids = model.tokenizer.encode(
    DOC, add_special_tokens=False
)[: model.max_seq_length - 2]
visible_text = model.tokenizer.decode(visible_ids)
print("\nText a single embedding of DOC actually sees (end of it):")
print("...", visible_text[-200:])

# ------------------------------------------------------------- Chunkers
def chunk_fixed(text: str, size: int, overlap: int) -> list[str]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def chunk_paragraphs(text: str, max_chars: int, overlap_sentences: int) -> list[str]:
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")
    if overlap_sentences < 0:
        raise ValueError("overlap_sentences must not be negative")

    chunks = []
    paragraphs = re.split(r"\n\s*\n", text.strip())

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= max_chars:
            chunks.append(paragraph)
            continue

        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", paragraph)
            if s.strip()
        ]
        current = []
        for sentence in sentences:
            if current and len(" ".join((*current, sentence))) > max_chars:
                chunks.append(" ".join(current))
                current = current[-overlap_sentences:] if overlap_sentences else []
            current.append(sentence)

        if current:
            chunks.append(" ".join(current))

    return chunks

def chunk_by_section(text, max_chars, overlap_sentences):
    parts = [p.strip() for p in re.split(r"\n\s*\n", text.strip())]
    sections = {}                       # heading -> list of paragraphs (keeps order)
    heading = "none"
    for p in parts:
        if len(p) < 60 and not re.search(r"[.!?]$", p):
            heading = p                 # a new section starts
        else:
            sections.setdefault(heading, []).append(p)

    out = []                            # list of (chunk_text, section)
    for heading, paras in sections.items():
        prefix = f"{heading}\n" if heading != "none" else ""
        budget = max_chars - len(prefix)
        for c in chunk_paragraphs("\n\n".join(paras), budget, overlap_sentences):
            out.append((prefix + c, heading))
    return out

pairs = chunk_by_section(DOC, 300, 1)
chunks = [c for c, _ in pairs]
metadatas = [{"section": s, "chunk_index": i} for i, (_, s) in enumerate(pairs)]


def attach_headings(text: str) -> str:
    parts = [p.strip() for p in re.split(r"\n\s*\n", text.strip())]
    out, heading = [], ""
    for p in parts:
        if len(p) < 60 and not re.search(r"[.!?]$", p):
            heading = p                      # a new section starts
        else:
            out.append(f"{heading}\n{p}" if heading else p)
    return "\n\n".join(out)


# --------------------------------------------------------------- Search
def make_index(chunks: list[str]):
    vecs = model.encode(chunks, normalize_embeddings=True)
    return chunks, vecs


def search(query: str, index, k: int = 3):
    chunks, vecs = index
    q = model.encode([query], normalize_embeddings=True)[0]
    sims = vecs @ q  # cosine, because all vectors are unit length
    top = np.argsort(sims)[::-1][:k]
    return [(float(sims[i]), chunks[i]) for i in top]


# -------------------------------------------------------------- Indexes
DOC_WITH_HEADINGS = attach_headings(DOC) 

# chunks, vecs = make_index(chunk_paragraphs(DOC_WITH_HEADINGS, 300, 1))  there is a bug with
#  this the header is not attached to the paragraphs that are long

chunks, vecs = make_index(
    [chunk for chunk, _ in chunk_by_section(DOC, 300, 1)]
)


import chromadb

client = chromadb.PersistentClient(path="./chroma_db")   # saved on disk


try:
    client.delete_collection("shop_docs")
except Exception:
    pass  # didn't exist yet
collection = client.get_or_create_collection(
    name="shop_docs", metadata={"hnsw:space": "cosine"}
)

headers = []
available_header = ["Returns and Refunds", "Shipping", "Warranty", "Accounts"]
for i, c in enumerate(chunks):
    parts = c.split("\n", 1)
    if len(parts) > 1 and parts[0] in available_header:
        headers.append({"section": parts[0], "chunk_index": i})
    else:
        headers.append({"section": "none", "chunk_index": i})

collection.upsert(
    ids=[f"c{i}" for i in range(len(chunks))],
    documents=chunks,
    embeddings=vecs.tolist(),
    metadatas=headers,
)

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

indexes = {
    # "fixed 300/0": make_index(chunk_fixed(DOC, size=300, overlap=0)),
    # "fixed 200/1 (your values)": make_index(chunk_fixed(DOC, size=200, overlap=1)),
    # "paragraph 300/1": make_index(chunk_paragraphs(DOC, 300, 1)),
    # "paragraph 200/1 (your values)": make_index(chunk_paragraphs(DOC, 200, 1)),
    # "paragraph+heading 300/1": make_index(chunk_paragraphs(DOC_WITH_HEADINGS, 300, 1)),
    "by_section 300/1 (same chunks as Chroma)": make_index(chunks),
    # "paragraph 900/1": make_index(chunk_paragraphs(DOC, 900, 1)),
    # "whole doc": make_index([DOC]),
}

queries = [
    "How long do I have to return an item?",
    "What isn't covered by the warranty?",
    # "Do I have to pay customs fees?",
    # "How do I reset my password?",       # my addition: answer is at the END of DOC
    "Who won the football world cup?",   # nothing relevant in the document
]

# # ----------------------------------------------------- Run everything
for name, idx in indexes.items():
    print(f"\n=== {name}: {len(idx[0])} chunks")
    for q in queries:
        score, text = search(q, idx, k=1)[0]
        print(f"  {score:.4f} | {q}")
        print(f"           -> {text[:90]!r}")


print(f"\n=== Chroma DB: {len(chunks)} chunks built, {collection.count()} stored")

for q in queries:
    score, text, meta = chroma_search(q)[0]
    print(f"  {score:.4f} | {q}")
    print(f"           -> {text[:90]!r}")

print(f"\n=== Chroma DB with section")

warranty_query = "What isn't covered by the warranty?"

for section in [None, "Warranty", "Returns and Refunds"]:
    results = chroma_search(warranty_query, k=5, section=section)
    print(f"\nsection={section!r}: {len(results)} result(s)")
    for score, text, meta in results:
        print(f"  {score:.4f} | {meta['section']} | {text[:60]!r}")

for m in collection.get()["metadatas"]:
    print(f"  {m['section']}")

# query = "What are the conditions for returning something?"

# without_heading = "Items must be unused and in their original packaging."
# with_heading = "Returns and Refunds\nItems must be unused and in their original packaging."

# q, a, b = model.encode([query, without_heading, with_heading], normalize_embeddings=True)

# print("without heading:", float(a @ q))
# print("with heading:   ", float(b @ q))

# ----------------------------------------- Optional: see all chunks
# for name in ["fixed 300/0", "paragraph+heading 300/1"]:
#     print(f"\n##### chunks of {name}")
#     for i, c in enumerate(indexes[name][0]):
#         print(f"[{i}] ({len(c)} chars) {c!r}")