import re
import numpy as np

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

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

print(model.max_seq_length)
print(len(model.tokenizer.encode(DOC)))

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
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", paragraph)
            if sentence.strip()
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

def embed(texts: list[str]) -> np.ndarray:
    return model.encode(texts, normalize_embeddings=False)

def chunk_fixed(text, size, overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        start += size - overlap
    return chunks

chunks_fixed = chunk_fixed(DOC, size=200, overlap=1)
chunks_paragraph = chunk_paragraphs(DOC, max_chars=200, overlap_sentences=1)
# for chunk in chunks_paragraph:
#     print(f"chunk: {chunk}")

embeddings_paragraph = embed(chunks_paragraph)

embeddings_fixed = embed(chunks_fixed)


def search_paragraphs(query, embeddings, chunks, k=3):
    quer_embed = embed([query])[0]
    sims = embeddings @ quer_embed
    top_k_indices = np.argsort(sims)[-k:][::-1]
    return [(chunks[i], sims[i]) for i in top_k_indices]


res = search_paragraphs("How long do I have to return an item?", embeddings_paragraph, chunks_paragraph)

print("result of search paragraphs: ")
for text, score in res:
    print(f"  {score:.4f}  {text}")

res_fix = search_paragraphs("How long do I have to return an item?", embeddings_fixed, chunks_fixed)

print("result of search fixed chunks: ")
for text, score in res_fix:
    print(f"  {score:.4f}  {text}")