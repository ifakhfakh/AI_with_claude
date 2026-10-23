import os
from openai import OpenAI

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
)

MODEL = "openai/gpt-oss-120b"
SYSTEM = {"role": "system", "content": "You are a concise assistant for backend developers."}

MODE = "summary"        # "none" | "window" | "summary"
WINDOW = 4              # messages kept verbatim (keep it even: user + assistant pairs)
SUMMARIZE_AFTER = 8     # summarize once history grows past this many messages

history = []            # user/assistant messages only; SYSTEM is added on every call
summary = ""            # compressed version of everything older than the window


def chat(messages):
    response = client.chat.completions.create(
        model=MODEL, temperature=0, max_tokens=1000, messages=messages
    )
    return response.choices[0].message.content, response.usage


def summarize(old_summary, old_messages):
    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in old_messages)
    prompt = (
        "Update the running summary of a conversation. Keep it under 80 words. "
        "Keep names, numbers, decisions, and constraints exactly as stated. "
        "Record only what the user stated or decided. Do not record questions or "
        "suggestions from the assistant as user facts. "
        "Drop small talk.\n\n"
        f"<previous_summary>{old_summary or 'none'}</previous_summary>\n\n"
        f"<new_messages>\n{transcript}\n</new_messages>"
    )
    text, _ = chat([{"role": "user", "content": prompt}])
    return text.strip()


def build_messages():
    messages = [SYSTEM]
    if MODE == "summary" and summary:
        messages.append({
            "role": "system",
            "content": f"Summary of the earlier conversation: {summary}",
        })
    if MODE == "none":
        messages += history
    else:
        messages += history[-WINDOW:] if MODE == "window" else history
    return messages


def maybe_summarize():
    global history, summary
    if MODE == "summary" and len(history) > SUMMARIZE_AFTER:
        old, history = history[:-WINDOW], history[-WINDOW:]
        summary = summarize(summary, old)
        print(f"\n[summary updated]: {summary}\n")


while True:
    user_input = input("You: ").strip()
    if not user_input:
        continue
    if user_input.lower() in ("quit", "exit", "q"):
        break

    history.append({"role": "user", "content": user_input})
    reply, usage = chat(build_messages())
    history.append({"role": "assistant", "content": reply})

    print(f"Bot: {reply}")
    print(f"[mode: {MODE} | prompt_tokens: {usage.prompt_tokens} | history: {len(history)} msgs]\n")

    maybe_summarize()