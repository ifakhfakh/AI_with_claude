import os
from openai import OpenAI

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
)

MODEL = "openai/gpt-oss-120b"

NOTE = """Weekly sync, Tuesday.
Sara said the payment API is still returning 500 errors on retries and she will
investigate before Thursday. Omar will update the onboarding docs after the
release, no date set. Lina has to confirm the staging server budget with
finance by the end of the month. Everyone agreed the demo on the 25th is the
top priority. Omar also mentioned he might look at the slow search page if he
has time."""


def call(system: str, user: str):
    """One model call. Returns (text, total_tokens)."""
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        max_tokens=1500,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content, response.usage.total_tokens


# ---------- Version 1: one call ----------

ONE_CALL_SYSTEM = (
    "You read meeting notes. Find every action item, then write a one-line "
    "status update for each owner. Use only what the notes say; never invent "
    "owners or dates. Output the action items as a list, then the status lines."
)

out1, tokens1 = call(ONE_CALL_SYSTEM, f"<notes>{NOTE}</notes>")

print("=" * 60)
print("ONE CALL")
print("=" * 60)
print(out1)
print(f"\n[total tokens: {tokens1}]")


# ---------- Version 2: two chained calls ----------

STEP1_SYSTEM = (
    "Extract every action item from the meeting notes. One per line in the "
    "format: owner | task | deadline (write 'none' if no deadline is stated). "
    "Use only what the notes say; never invent owners or dates. If something "
    "is only a possibility, not a commitment, mark it as '(tentative)'."
)

STEP2_SYSTEM = (
    "You are given a list of action items in the format owner | task | "
    "deadline. Write one short status-update line per owner, in plain "
    "language. Do not add information that is not in the list."
)

items, tokens_a = call(STEP1_SYSTEM, f"<notes>{NOTE}</notes>")
status, tokens_b = call(STEP2_SYSTEM, items)

print("\n" + "=" * 60)
print("TWO CALLS")
print("=" * 60)
print("STEP 1 (extracted items):")
print(items)
print("\nSTEP 2 (status updates):")
print(status)
print(f"\n[tokens: step 1 = {tokens_a}, step 2 = {tokens_b}, total = {tokens_a + tokens_b}]")

print("\n" + "=" * 60)
print(f"COMPARISON: one call = {tokens1} tokens, two calls = {tokens_a + tokens_b} tokens")