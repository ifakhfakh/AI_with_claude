from typing import Literal
import json
import os
from openai import OpenAI

from pydantic import BaseModel, ConfigDict

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
)

old_prompt = "You are a concise assistant that returns structured "
"json response containing a task, deadline, and priority. The task is a string, the deadline is an optional string, "

new_prompt = (
    "Extract every distinct task from the message as a separate item in the "
    "tasks list. If one sentence contains several actions (for example, joined "
    "by 'and'), return one item per action. If the message contains no task, "
    "return an empty list. The deadline is a string, or null if none is stated; "
    "never invent one. Priority is low, medium, or high based on urgency. "
    "If a single deadline is stated for a group of tasks, apply it to every task in that group"
)


SYSTEM = {"role": "system", "content": new_prompt}


class Task(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str
    deadline: str | None
    priority: Literal["low", "medium", "high"]

class TaskList(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tasks: list[Task]

two_tasks = "Fix the login bug and write a report by Friday, it's urgent"
zero_task="hello Joe how is it going ?"
response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        temperature=0,
        max_tokens=1000,
        messages=[SYSTEM] + [{"role": "user", "content": zero_task}],
        response_format={
        "type": "json_schema",
        "json_schema": {
            "strict": True,
            "name": "task_schema",
            "schema": TaskList.model_json_schema()
        }
    }
    )


reply = response.choices[0].message.content
try: 
    task_list = TaskList.model_validate_json(reply)  # This will raise a ValidationError if the response is not valid JSON or doesn't match the model
except Exception as e:
    print(f"Error parsing response: {e}")
    task = None

print(f"Bot: {reply}")