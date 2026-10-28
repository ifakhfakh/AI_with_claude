import json
import os
from openai import OpenAI

# -------------------------------------------------------------- retrieval


llm = OpenAI(base_url="https://api.groq.com/openai/v1",
                api_key=os.environ["GROQ_API_KEY"])
MODEL = "openai/gpt-oss-120b"
MAX_ROUNDS = 4
MAX_TOKENS = 2000          # reasoning tokens count, keep headroom

SYSTEM = """ Only use the tool when the question is related to Returns and Refunds, Shipping, Warranty and Accounts. 
IF the user is just greeting you just respond directly without using tool. """         # GAP 1: write it (see thinking question 1)

TOOLS = [{
    "type": "function",
    "function": {
        "name": "search_docs",
        "description": "Searches through a collection of documents and returns relevant snippets.",          # GAP 2: what it searches, when to use it
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "question to give answer to"},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}]

import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="/workspaces/AI_with_claude/lesson10_chromadb/chroma_db")
collection = client.get_collection("shop_docs")   # raises if it doesn't exist

def retrieve(question: str, k: int = 3) -> list[dict]:
    query_vec = model.encode([question], normalize_embeddings=True)[0].tolist()
    res = collection.query(query_embeddings=[query_vec], n_results=k)
    return [
        {"id": cid, "text": doc, "section": meta["section"], "score": 1 - dist}
        for cid, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    ]


def search_docs(query: str, k: int = 3) -> str:
    chunks = retrieve(query, k)
    blocks = [
        f'<chunk id="{c["id"]}" section="{c["section"]}">\n{c["text"]}\n</chunk>'
        for c in chunks
    ]
    context = "\n".join(blocks)
    return f"<context>\n{context}\n</context>\n"

available_functions = {
    "search_docs": search_docs
}

def execute_tool(name: str, args: dict) -> str:
    # print(call.function.name, call.function.arguments)
    # name = call.function.name
    if name not in available_functions:
        return f"Error: unknown tool '{name}'. Available: {list(available_functions)}"
    try:
        args_json = json.loads(args)
    except json.JSONDecodeError:
        return "Error: arguments were not valid JSON."
    try:
        return str(available_functions[name](**args_json))
    except Exception as e:
        return f"Error: {name} failed with {type(e).__name__}: {e}"



def run_agent(question: str) -> dict:
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": question}]
    trace = {"rounds": 0, "tool_calls": [], "completion_tokens": 0}

    for round_no in range(MAX_ROUNDS):
        resp = llm.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS, max_tokens=MAX_TOKENS,
        )
        
        trace["rounds"] += 1
        trace["completion_tokens"] += resp.usage.completion_tokens
        choice = resp.choices[0]
        msg = choice.message
        messages.append(msg)
        if(choice.finish_reason == "length"):
            print({"answer": "Output was cut off (increase max_tokens?)"})
            return {"answer": msg.content, **trace}

        # GAP 4: handle finish_reason == "length" (what should happen, and what
        # should the caller see?)
        for call in msg.tool_calls:
            trace["tool_calls"].append({"name": call.function.name, "args": call.function.arguments})
            tool_result = execute_tool(call.function.name, call.function.arguments)
            messages.append({"role": "tool",  "tool_call_id": call.id, "content": tool_result})

        if not msg.tool_calls:
            return {"answer": msg.content, **trace}

        # GAP 5: append the assistant message, run every tool call, append
        # each result back, and record (name, args) in trace["tool_calls"]

    # GAP 6: the round cap was hit without a final answer. Return something
    # honest instead of None.


if __name__ == "__main__":
    test_questions = [
        "What is the return policy for electronics?",
        "How long does shipping take for international orders?",
        "Is there a warranty on the new smartphone model?",
        "I just wanted to say hello!",
        "Can you help me with my account login issue?",
    ]
    result = run_agent(test_questions[0])
    print(result)
    ...  # GAP 7: loop over your test questions, print answer + trace