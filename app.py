import os
from openai import OpenAI
import time

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
)

SYSTEM = {"role": "system", "content": "You are a concise assistant for backend developers."}
history = []

def get_weather(city: str) -> str:
    return f"Sunny, 24°C in {city}"  # fake, for learning

def add_numbers(a: int, b: int) -> int:
    return a + b

def get_user_city(user_id: int) -> str:
    cities = {42: "Tunis", 7: "Paris"}
    return cities.get(user_id, "unknown")

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}, {
    "type": "function",
    "function": {
        "name": "add_numbers",
        "description": "Add two numbers",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    },
}, {
    "type": "function",
    "function": {
        "name": "get_user_city",
        "description": "Get the city of a user",
        "parameters": {
            "type": "object",
            "properties": {"user_id": {"type": "integer"}},
            "required": ["user_id"],
        },
    },
}]

available_functions = {
    "add_numbers": add_numbers,
    "get_weather": get_weather,
    "get_user_city": get_user_city,
}


def get_streaming_response(messages, start_time):
    stream = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        temperature=0,
        messages=messages,
        max_tokens=1000,
        stream=True,
        tools=tools
    )

    full_reply = ""
    for chunk in stream:
        if not chunk.choices:
            continue
        piece = chunk.choices[0].delta.tool_calls
        if piece:
            if not full_reply:
                print(
                    f"\n[streaming response, first chunk received in "
                    f"{time.perf_counter() - start_time:.2f} seconds]\n"
                )
            print(piece, end="", flush=True)
            # full_reply += piece

    total_end_time = time.perf_counter() - start_time
    print(f"\n[streaming response completed in {total_end_time:.2f} seconds]\n")
    return full_reply


def get_non_streaming_response(messages, start_time):
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        temperature=0,
        messages=messages,
        max_tokens=1000,
        stream=False,
    )
    if not response.choices:
        raise RuntimeError("The API returned no completion choices.")

    reply = response.choices[0].message.content
    if reply is None:
        raise RuntimeError("The API returned a completion without text content.")

    print(f"Bot: {reply}")
    total_end_time = time.perf_counter() - start_time
    print(f"\n[response completed in {total_end_time:.2f} seconds]\n")
    return reply

while True:
    user_input = input("You: ").strip()
    if not user_input:
        continue
    if user_input.lower() in ("quit", "exit", "q"):
        print("Goodbye!")
        break

    history.append({"role": "user", "content": user_input})

    start_time = time.perf_counter()

    # messages = [SYSTEM] + history
    messages = history
    reply = get_streaming_response(messages, start_time)
    # Use get_non_streaming_response(messages, start_time) to disable streaming.
    history.append({"role": "assistant", "content": reply})
    print(reply)
    print()
