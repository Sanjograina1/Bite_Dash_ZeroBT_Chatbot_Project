from openai import OpenAI

# 🚨 Replace with your actual OpenAI API key (starts with sk-...)
OPENAI_KEY = "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"

client = OpenAI(api_key=OPENAI_KEY)

print("\n🚀 Sending request to OpenAI API...")

response = client.chat.completions.create(
    model="gpt-4o-mini",  # or "gpt-4o" / "gpt-3.5-turbo"
    messages=[
        {"role": "system", "content": "You are a helpful customer support bot."},
        {"role": "user", "content": "Explain RAG architecture in 2 sentences."}
    ]
)

print("\n--- API RESPONSE ---")
print(response.choices[0].message.content)

print("\n--- TOKEN USAGE ---")
print("Prompt Tokens:", response.usage.prompt_tokens)
print("Completion Tokens:", response.usage.completion_tokens)
print("Total Tokens Consumed:", response.usage.total_tokens)