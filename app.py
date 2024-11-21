import os
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)

# Define model mappings for supported languages
MODELS = {
    "hindi": "Helsinki-NLP/opus-mt-en-hi",  # English to Hindi model
    "french": "Helsinki-NLP/opus-mt-en-fr",  # English to French model
}

@app.route('/translate', methods=['POST'])
def translate_text():
    data = request.get_json()

    # Get the text to translate and target language
    text = data.get("text", "Hello, how are you?")
    target_language = data.get("language", "hindi").lower()

    if target_language not in MODELS:
        return jsonify({"error": f"Unsupported language '{target_language}'. Supported languages are 'hindi' and 'french'."}), 400

    try:
        # Dynamically load the model and tokenizer for the requested language
        model_name = MODELS[target_language]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Tokenize and generate translation
        inputs = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(inputs, max_length=512, num_beams=5, early_stopping=True)
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return jsonify({"translation": translated_text, "language": target_language})
    except Exception as e:
        return jsonify({"error": "An error occurred while processing the translation.", "details": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5001)))

