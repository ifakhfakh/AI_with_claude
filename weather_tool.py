import json
from typing import Literal
import json
import os
from openai import OpenAI

from pydantic import BaseModel, ConfigDict

def get_user_city(user_id: int) -> str:
    # raise NotImplementedError("Error: get_user_city failed with RuntimeError: database timeout. This may be temporary; retrying is safe.")
    cities = {42: "Tunis", 7: "Paris"}
    return cities.get(user_id, "unknown")

def get_weather(city: str) -> str:
    return f"Sunny, 24°C in {city}"  # fake, for learning

def add_numbers(a: int, b: int) -> int:
    return a + b

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


client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
)

system_message = {"role": "system", "content": 
                  "Only use the provided tools. If a request needs an action you have no tool for, say so plainly. "
                  "Use your tools for anything they can do, and say clearly what you can’t do"}

weather_prompt = "What's the weather in Tunis?"
add_numbers_promptt = "Add 5 and 7"
double_tool = "What’s the weather in Tunis, and what’s 5 plus 7?"
non_int_add = "Add 2.5 and 3"
messages = [system_message, {"role": "user", "content": "send an email to Sam where user 42 lives?"}]
response = client.chat.completions.create(
    model="openai/gpt-oss-120b", messages=messages, tools=tools
)

# print(response.choices[0].message)

msg = response.choices[0].message

limit = 0

def execute_tool(call):
    print(call.function.name, call.function.arguments)
    name = call.function.name
    if name not in available_functions:
        return f"Error: unknown tool '{name}'. Available: {list(available_functions)}"
    try:
        args = json.loads(call.function.arguments)
    except json.JSONDecodeError:
        return "Error: arguments were not valid JSON."
    try:
        return str(available_functions[name](**args))
    except Exception as e:
        return f"Error: {name} failed with {type(e).__name__}: {e}"

while msg.tool_calls and limit < 4:  # Prevent infinite loops
    limit += 1
    messages.append(msg)  # keep the model's request in history
    for call in msg.tool_calls:
        result = execute_tool( call)
        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": result,
        })
    final = client.chat.completions.create(
        model="openai/gpt-oss-120b", messages=messages, tools=tools
    )
    msg = final.choices[0].message

if msg.tool_calls:
    print("Stopped: too many tool rounds.")

if (limit >0):
    print("final tooling result:", msg.content)
else:
    print("final result:", msg.content)