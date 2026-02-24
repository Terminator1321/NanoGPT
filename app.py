import os
import torch
from flask import Flask, render_template, request, jsonify, session
from NanoGPT import NanoGPT as gpt
from NanoGPT import generate, FeedBackTraning
from tokenizers import Tokenizer

app = Flask(__name__)
app.secret_key = "nano_gpt_secret_key"

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = Tokenizer.from_file("bpe_tokenizer.json")

MODEL_PATHS = {
    "v1": "finishedVersionModels/NanoGPT_v1.pt",
    "v2": "finishedVersionModels/NanoGPT_v2.pt",
    "v3": "finishedVersionModels/NanoGPT_v3.pt",
    "v4": "finishedVersionModels/NanoGPT_v4_feedback.pt",
}

current_model_key = "v3"

model = None
feedback_system = None

def load_model(key):
    global model, feedback_system, current_model_key

    if key not in MODEL_PATHS:
        return False

    path = MODEL_PATHS[key]

    if not os.path.exists(path):
        return False

    # Recreate model cleanly
    model = gpt().to(device)
    model.load(path)
    model.eval()

    feedback_system = FeedBackTraning(model, tokenizer, device)

    current_model_key = key
    print(f"Loaded model: {key}")
    return True

load_model(current_model_key)
@app.route("/")
def index():
    session["chat_history"] = []
    return render_template("index.html",models=list(MODEL_PATHS.keys()),current=current_model_key)


@app.route("/switch_model", methods=["POST"])
def switch_model():
    data = request.get_json()

    if not data or "model" not in data:
        return jsonify({"status": "Invalid request"}), 400

    key = data["model"]

    if load_model(key):
        session["chat_history"] = []
        return jsonify({"status": f"Switched to {key}"})
    else:
        return jsonify({"status": "Model not found"}), 404


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()

    if not data or "message" not in data:
        return jsonify({"status": "Invalid request"}), 400

    user_input = data["message"]

    prompt = f"<USER>\n{user_input}\n<AI>\n"
    encoded = tokenizer.encode(prompt).ids
    input_len = len(encoded)

    start = torch.tensor([encoded], dtype=torch.long, device=device)
    end_token_id = tokenizer.token_to_id("<END>")

    generated = generate(model,start,max_new_tokens=120,stop_token_id=end_token_id,temperature=0.7,top_k=30 )

    new_tokens = generated[0][input_len:]
    decoded = tokenizer.decode(new_tokens.tolist())
    ai_response = decoded.replace("<END>", "").strip()

    history = session.get("chat_history", [])
    history.append({"role": "user", "content": user_input})
    history.append({"role": "ai", "content": ai_response})
    session["chat_history"] = history

    return jsonify({"response": ai_response})


@app.route("/chat_history")
def chat_history():
    return jsonify(session.get("chat_history", []))


@app.route("/fix", methods=["POST"])
def fix():
    data = request.get_json()

    if not data or "correction" not in data:
        return jsonify({"status": "Invalid request"}), 400

    correction = data["correction"]

    history = session.get("chat_history", [])
    if len(history) < 2:
        return jsonify({"status": "Nothing to fix"}), 400

    last_user_input = history[-2]["content"]
    prompt = f"<USER>\n{last_user_input}\n<AI>\n"

    feedback_system.save_feedback(prompt, correction)# type: ignore

    return jsonify({"status": "Correction saved."})


@app.route("/train", methods=["POST"])
def train():
    feedback_system.train_on_feedback(epochs=2) # type: ignore
    return jsonify({"status": "Training complete."})


@app.route("/save", methods=["POST"])
def save():
    path = MODEL_PATHS["v4"]

    feedback_system.save_model(path)# type: ignore

    load_model("v4")

    return jsonify({"status": "Saved and reloaded v4."})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)