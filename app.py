import os
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)

# Load BioGPT model
model_name = "microsoft/BioGPT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

@app.route('/getAdvice', methods=['POST'])
def get_advice():
    data = request.get_json()
    age = data.get("age", 12)  # Default age
    name = data.get("name", "Gurpinder")  # Default name
    behavior = data.get("behavior", "smoking")  # Default behavior

    # Construct the input prompt
    input_prompt = (
        f"As a health advisor, provide actionable health advice for {name}, "
        f"a {age}-year-old who is involved in {behavior}. Focus on specific tips for diet, "
        f"exercise, and stress management. Ensure the advice is clear and evidence-based."
    )

    # Tokenize the input and generate advice
    inputs = tokenizer.encode(input_prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1, do_sample=True, temperature=0.7)

    # Decode the generated text
    advice = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"advice": advice})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5001)))
