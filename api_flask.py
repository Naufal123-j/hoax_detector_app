from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import utils.preprocessing as prep

app = Flask(__name__)
CORS(app)

tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
model = BertForSequenceClassification.from_pretrained('./model')
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get("text", "")
    clean = prep.clean_text(text)
    inputs = tokenizer(clean, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs).item()
    label = "hoaks" if pred == 1 else "fakta"
    return jsonify({"prediction": label, "confidence": round(probs[0][pred].item(), 2)})

if __name__ == '__main__':
    app.run(debug=True)
