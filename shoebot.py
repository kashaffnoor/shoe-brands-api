"""ShoeBot - Python version of the n8n multi-brand shoe AI agent.

Same design as the n8n workflow:
  - 5 brand tools that GET each brand's JSON (HTTP Request tools in n8n)
  - Groq LLM (OpenAI-compatible API) with tool calling (AI Agent in n8n)
  - System message + chat memory (Simple Memory in n8n)

Run:
  pip install requests openai
  export GROQ_API_KEY="gsk_..."        (Windows: set GROQ_API_KEY=gsk_...)
  python shoebot.py
"""
import json
import os

import requests
from openai import OpenAI

BASE = "https://raw.githubusercontent.com/kashaffnoor/shoe-brands-api/main/"
BRANDS = ["Nike", "Adidas", "Puma", "Bata", "Servis"]
MODEL = "openai/gpt-oss-120b"

SYSTEM_MESSAGE = """You are "ShoeBot", a customer support AI agent for five shoe brands: Nike, Adidas, Puma, Bata and Servis.

Rules:
1. Always call the relevant brand tool (Nike_API, Adidas_API, Puma_API, Bata_API, Servis_API) before answering. Never guess prices or stock.
2. For comparisons between brands, call every brand tool involved.
3. Prices are in PKR. Mention size availability and stock status.
4. If a product is out of stock, say so and suggest a similar in-stock product.
5. If the user asks about a brand outside these five, politely say you only support these five brands.
6. Keep answers short, friendly and clear."""


def fetch_brand(brand):
    """HTTP GET request to one brand's data endpoint (same as an n8n HTTP Request tool)."""
    r = requests.get(BASE + brand.lower() + ".json", timeout=20)
    r.raise_for_status()
    return r.json()


# One tool definition per brand, same names and descriptions as the n8n nodes
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": f"{b}_API",
            "description": f"Use this tool to get {b} shoe products, prices in PKR, sizes, "
                           f"colors, stock availability, return policy and delivery info.",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    for b in BRANDS
]


def run_tool(name):
    brand = name.replace("_API", "")
    if brand not in BRANDS:
        return {"error": f"Unknown tool {name}"}
    try:
        return fetch_brand(brand)
    except requests.RequestException as e:
        return {"error": f"Information from {brand} is currently unavailable ({e})"}


def ask_agent(client, history, user_msg, max_steps=6):
    """Agent loop: model decides which tools to call, we run them, model writes the answer."""
    history.append({"role": "user", "content": user_msg})
    for _ in range(max_steps):
        resp = client.chat.completions.create(model=MODEL, messages=history, tools=TOOLS)
        msg = resp.choices[0].message
        if not msg.tool_calls:
            history.append({"role": "assistant", "content": msg.content})
            return msg.content
        history.append(msg.model_dump(exclude_none=True))
        for call in msg.tool_calls:
            print(f"   [tool called: {call.function.name}]")
            result = run_tool(call.function.name)
            history.append({"role": "tool", "tool_call_id": call.id,
                            "content": json.dumps(result)})
    return "Sorry, I could not complete that request."


def main():
    client = OpenAI(api_key=os.environ["GROQ_API_KEY"],
                    base_url="https://api.groq.com/openai/v1")
    history = [{"role": "system", "content": SYSTEM_MESSAGE}]
    print("ShoeBot ready. Ask about Nike, Adidas, Puma, Bata or Servis (type 'exit' to quit).")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        if q:
            print("ShoeBot:", ask_agent(client, history, q))
            # keep memory small, like Simple Memory (system + last 10 messages)
            if len(history) > 11:
                history[:] = [history[0]] + history[-10:]


if __name__ == "__main__":
    main()
