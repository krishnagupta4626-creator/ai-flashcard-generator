import json
import re
from huggingface_hub import InferenceClient

# Initialize client (uses your local Hugging Face login automatically)
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
client = InferenceClient(model=MODEL_ID)



# 3. Input study text
source_text = """
Loops are used to execute a block of code repeatedly until a condition is met or all items in a sequence are processed. 
The main types are For loops (iterating over sequences) and While loops (executing code based on a condition).
For loops are used to iterate over sequences such as a list, tuple, string, or range.
"""

num_cards = 3

prompt = f"""
You are an educational assistant. Read the provided text and extract {num_cards} key flashcards.

Text:
"{source_text}"

Respond ONLY with a raw JSON array of objects. Do not include Markdown formatting, code blocks, or extra text.
Each object in the array must have two keys: "question" and "answer".

Example output format:
[
    {{"question": "What is photosynthesis?", "answer": "The process by which plants convert light energy into chemical energy."}}
]
"""

print("Sending request to Hugging Face Inference API...\n")

try:
    # 4. Perform direct chat completion request
    response = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0.3
    )
    
    # Extract generated output text
    raw_content = response.choices[0].message.content.strip()

    # Clean potential JSON code-block wrapping
    clean_json = re.sub(r"^```(json)?|```$", "", raw_content, flags=re.MULTILINE).strip()
    
    # Parse JSON
    cards = json.loads(clean_json)

    # 5. Output flashcards in terminal
    print("=" * 50)
    print("           GENERATED FLASHCARDS")
    print("=" * 50)
    for idx, card in enumerate(cards, start=1):
        print(f"\n[Card {idx}]")
        print(f"Q: {card.get('question')}")
        print(f"A: {card.get('answer')}")
        print("-" * 50)

except Exception as e:
    print(f"Error during API call: {e}")