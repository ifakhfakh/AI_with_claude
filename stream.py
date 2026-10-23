stream = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[{"role": "user", "content": "Explain idempotency in 5 sentences."}],
    max_tokens=1000,
    stream=True,
)

full_reply = ""
for chunk in stream:
    if not chunk.choices:
        continue
    piece = chunk.choices[0].delta.content
    if piece:
        print(piece, end="", flush=True)
        full_reply += piece
print()